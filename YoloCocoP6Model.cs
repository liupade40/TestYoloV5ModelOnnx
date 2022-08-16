using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Yolov5Net.Scorer;
using Yolov5Net.Scorer.Models.Abstract;

namespace TestYoloV5ModelOnnx
{
    public class YoloCocoP6Model : YoloModel
    {
        public override int Width { get; set; } = 640;


        public override int Height { get; set; } = 640;


        public override int Depth { get; set; } = 3;


        public override int Dimensions { get; set; } = 23;


        public override int[] Strides { get; set; } = new int[3] { 8, 16, 32 };


        public override int[][][] Anchors { get; set; } = new int[3][][]
        {
            new int[3][]
            {
                new int[2] { 10, 13 },
                new int[2] { 16, 30 },
                new int[2] { 33, 23 }
            },
            new int[3][]
            {
                new int[2] { 30, 61 },
                new int[2] { 62, 45 },
                new int[2] { 59, 119 }
            },
            new int[3][]
            {
                new int[2] { 116, 90 },
                new int[2] { 156, 198 },
                new int[2] { 373, 326 }
            }
        };


        public override int[] Shapes { get; set; } = new int[3] { 80, 40, 20 };


        public override float Confidence { get; set; } = 0.2f;


        public override float MulConfidence { get; set; } = 0.25f;


        public override float Overlap { get; set; } = 0.45f;


        public override string[] Outputs { get; set; } = new string[1] { "output" };


        public override List<YoloLabel> Labels { get; set; } = new List<YoloLabel>
        {
             
        };


        public override bool UseDetect { get; set; } = true;

    }
}
