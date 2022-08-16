using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Yolov5Net.Scorer;
using Yolov5Net.Scorer.Extensions;
using Yolov5Net.Scorer.Models.Abstract;

namespace TestYoloV5ModelOnnx
{
    public class YoloScorerV2<T> : IDisposable where T : YoloModel
    {
        private readonly YoloCocoP6Model _model;

        private readonly InferenceSession _inferenceSession;

        private float Sigmoid(float value)
        {
            return 1f / (1f + (float)Math.Exp(0f - value));
        }

        private float[] Xywh2xyxy(float[] source)
        {
            return new float[4]
            {
                source[0] - source[2] / 2f,
                source[1] - source[3] / 2f,
                source[0] + source[2] / 2f,
                source[1] + source[3] / 2f
            };
        }

        public float Clamp(float value, float min, float max)
        {
            if (!(value < min))
            {
                if (!(value > max))
                {
                    return value;
                }

                return max;
            }

            return min;
        }

        private Bitmap ResizeImage(Image image)
        {
            PixelFormat pixelFormat = image.PixelFormat;
            Bitmap bitmap = new Bitmap(_model.Width, _model.Height, pixelFormat);
            int width = image.Width;
            int height = image.Height;
            int num = width;
            int num2 = height;
            float num3 = (float)_model.Width / (float)num;
            float num4 = (float)_model.Height / (float)num2;
            float val = num3;
            float val2 = num4;
            float num5 = Math.Min(val, val2);
            width = (int)((float)num * num5);
            int num6 = (int)((float)num2 * num5);
            int num7 = width;
            int num8 = num6;
            width = _model.Width / 2 - num7 / 2;
            int num9 = _model.Height / 2 - num8 / 2;
            int x = width;
            int y = num9;
            Rectangle rect = new Rectangle(x, y, num7, num8);
            using Graphics graphics = Graphics.FromImage(bitmap);
            graphics.Clear(Color.FromArgb(0, 0, 0, 0));
            graphics.SmoothingMode = SmoothingMode.None;
            graphics.InterpolationMode = InterpolationMode.Bilinear;
            graphics.PixelOffsetMode = PixelOffsetMode.Half;
            graphics.DrawImage(image, rect);
            return bitmap;
        }

        private unsafe Tensor<float> ExtractPixels(Image image)
        {
            Bitmap bitmap = (Bitmap)image;
            Rectangle rect = new Rectangle(0, 0, bitmap.Width, bitmap.Height);
            BitmapData bitmapData = bitmap.LockBits(rect, ImageLockMode.ReadOnly, bitmap.PixelFormat);
            int bytesPerPixel = Image.GetPixelFormatSize(bitmap.PixelFormat) / 8;
            DenseTensor<float> tensor = new DenseTensor<float>(new int[4] { 1, 3, _model.Height, _model.Width });
            Parallel.For(0, bitmapData.Height, delegate (int y)
            {
                byte* row = (byte*)(void*)bitmapData.Scan0 + y * bitmapData.Stride;
                Parallel.For(0, bitmapData.Width, delegate (int x)
                {
                    tensor[new int[4] { 0, 0, y, x }] = (float)(int)row[x * bytesPerPixel + 2] / 255f;
                    tensor[new int[4] { 0, 1, y, x }] = (float)(int)row[x * bytesPerPixel + 1] / 255f;
                    tensor[new int[4] { 0, 2, y, x }] = (float)(int)row[x * bytesPerPixel] / 255f;
                });
            });
            bitmap.UnlockBits(bitmapData);
            return tensor;
        }

        private DenseTensor<float>[] Inference(Image image)
        {
            Bitmap bitmap = null;
            if (image.Width != _model.Width || image.Height != _model.Height)
            {
                bitmap = ResizeImage(image);
            }

            List<NamedOnnxValue> inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("images", ExtractPixels(bitmap ?? image)) };
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> source = _inferenceSession.Run(inputs);
            List<DenseTensor<float>> list = new List<DenseTensor<float>>();
            string[] outputs = _model.Outputs;
            foreach (string item in outputs)
            {
                list.Add(source.First((DisposableNamedOnnxValue x) => x.Name == item).Value as DenseTensor<float>);
            }

            return list.ToArray();
        }

        private List<YoloPrediction> ParseDetect(DenseTensor<float> output, Image image)
        {
            ConcurrentBag<YoloPrediction> result = new ConcurrentBag<YoloPrediction>();
            int width = image.Width;
            int height = image.Height;
            int w = width;
            int h = height;
            float num = (float)_model.Width / (float)w;
            float num2 = (float)_model.Height / (float)h;
            float val = num;
            float val2 = num2;
            float gain = Math.Min(val, val2);
            num = ((float)_model.Width - (float)w * gain) / 2f;
            float num3 = ((float)_model.Height - (float)h * gain) / 2f;
            float xPad = num;
            float yPad = num3;
            Parallel.For(0, (int)output.Length / _model.Dimensions, delegate (int i)
            {
                if (!(output[new int[3] { 0, i, 4 }] <= _model.Confidence))
                {
                    Parallel.For(5, _model.Dimensions, delegate (int j)
                    {
                        output[new int[3] { 0, i, j }] = output[new int[3] { 0, i, j }] * output[new int[3] { 0, i, 4 }];
                    });
                    Parallel.For(5, _model.Dimensions, delegate (int k)
                    {
                        if (!(output[new int[3] { 0, i, k }] <= _model.MulConfidence))
                        {
                            float value = (output[new int[3] { 0, i, 0 }] - output[new int[3] { 0, i, 2 }] / 2f - xPad) / gain;
                            float value2 = (output[new int[3] { 0, i, 1 }] - output[new int[3] { 0, i, 3 }] / 2f - yPad) / gain;
                            float value3 = (output[new int[3] { 0, i, 0 }] + output[new int[3] { 0, i, 2 }] / 2f - xPad) / gain;
                            float value4 = (output[new int[3] { 0, i, 1 }] + output[new int[3] { 0, i, 3 }] / 2f - yPad) / gain;
                            value = Clamp(value, 0f, w);
                            value2 = Clamp(value2, 0f, h);
                            value3 = Clamp(value3, 0f, w - 1);
                            value4 = Clamp(value4, 0f, h - 1);
                            YoloPrediction item = new YoloPrediction(_model.Labels[k - 5], output[new int[3] { 0, i, k }])
                            {
                                Rectangle = new RectangleF(value, value2, value3 - value, value4 - value2)
                            };
                            result.Add(item);
                        }
                    });
                }
            });
            return result.ToList();
        }

        private List<YoloPrediction> ParseSigmoid(DenseTensor<float>[] output, Image image)
        {
            ConcurrentBag<YoloPrediction> result = new ConcurrentBag<YoloPrediction>();
            int width = image.Width;
            int height = image.Height;
            int w = width;
            int h = height;
            float num = (float)_model.Width / (float)w;
            float num2 = (float)_model.Height / (float)h;
            float val = num;
            float val2 = num2;
            float gain = Math.Min(val, val2);
            num = ((float)_model.Width - (float)w * gain) / 2f;
            float num3 = ((float)_model.Height - (float)h * gain) / 2f;
            float xPad = num;
            float yPad = num3;
            Parallel.For(0, output.Length, delegate (int i)
            {
                int shapes = _model.Shapes[i];
                Parallel.For(0, _model.Anchors[0].Length, delegate (int a)
                {
                    Parallel.For(0, shapes, delegate (int y)
                    {
                        Parallel.For(0, shapes, delegate (int x)
                        {
                            int count = (shapes * shapes * a + shapes * y + x) * _model.Dimensions;
                            float[] buffer = output[i].Skip(count).Take(_model.Dimensions).Select(new Func<float, float>(Sigmoid))
                                .ToArray();
                            if (!(buffer[4] <= _model.Confidence))
                            {
                                List<float> list = (from b in buffer.Skip(5)
                                                    select b * buffer[4]).ToList();
                                float num4 = list.Max();
                                if (!(num4 <= _model.MulConfidence))
                                {
                                    float num5 = (buffer[0] * 2f - 0.5f + (float)x) * (float)_model.Strides[i];
                                    float num6 = (buffer[1] * 2f - 0.5f + (float)y) * (float)_model.Strides[i];
                                    float num7 = (float)Math.Pow(buffer[2] * 2f, 2.0) * (float)_model.Anchors[i][a][0];
                                    float num8 = (float)Math.Pow(buffer[3] * 2f, 2.0) * (float)_model.Anchors[i][a][1];
                                    float[] array = Xywh2xyxy(new float[4] { num5, num6, num7, num8 });
                                    float num9 = Clamp((array[0] - xPad) / gain, 0f, w);
                                    float num10 = Clamp((array[1] - yPad) / gain, 0f, h);
                                    float num11 = Clamp((array[2] - xPad) / gain, 0f, w - 1);
                                    float num12 = Clamp((array[3] - yPad) / gain, 0f, h - 1);
                                    YoloPrediction item = new YoloPrediction(_model.Labels[list.IndexOf(num4)], num4)
                                    {
                                        Rectangle = new RectangleF(num9, num10, num11 - num9, num12 - num10)
                                    };
                                    result.Add(item);
                                }
                            }
                        });
                    });
                });
            });
            return result.ToList();
        }

        private List<YoloPrediction> ParseOutput(DenseTensor<float>[] output, Image image)
        {
            if (!_model.UseDetect)
            {
                return ParseSigmoid(output, image);
            }

            return ParseDetect(output[0], image);
        }

        private List<YoloPrediction> Supress(List<YoloPrediction> items)
        {
            List<YoloPrediction> list = new List<YoloPrediction>(items);
            foreach (YoloPrediction item in items)
            {
                foreach (YoloPrediction item2 in list.ToList())
                {
                    if (item2 != item)
                    {
                        RectangleF rectangle = item.Rectangle;
                        RectangleF rectangle2 = item2.Rectangle;
                        RectangleF rectangleF = rectangle;
                        RectangleF rectangleF2 = rectangle2;
                        float num = RectangleF.Intersect(rectangleF, rectangleF2).Area();
                        float num2 = rectangleF.Area() + rectangleF2.Area() - num;
                        if (num / num2 >= _model.Overlap && item.Score >= item2.Score)
                        {
                            list.Remove(item2);
                        }
                    }
                }
            }

            return list;
        }

        public List<YoloPrediction> Predict(Image image)
        {
            return Supress(ParseOutput(Inference(image), image));
        }

        public YoloScorerV2(List<string> labels)
        {
            _model =new YoloCocoP6Model();
            _model.Dimensions = labels.Count + 5;
            for (int i = 0; i < labels.Count; i++)
            {
                _model.Labels.Add(new YoloLabel()
                {
                    Id = i,
                    Name = labels[i]
                });
            }
        }
        public YoloScorerV2(string weights,List<string> labels, SessionOptions opts = null)
            : this(labels)
        {
            _inferenceSession = new InferenceSession(File.ReadAllBytes(weights), opts ?? new SessionOptions());
        }
    

        public void Dispose()
        {
            _inferenceSession.Dispose();
        }
    }
}
