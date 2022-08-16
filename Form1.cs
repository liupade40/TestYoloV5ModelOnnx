using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Yolov5Net.Scorer;
using Yolov5Net.Scorer.Models;

namespace TestYoloV5ModelOnnx
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.InitialDirectory = AppDomain.CurrentDomain.BaseDirectory;
            ofd.Filter = "onnx文件|*.onnx";
            ofd.RestoreDirectory = true;
            var show = ofd.ShowDialog();
            if (show == DialogResult.OK)
            {
                textBox1.Text = ofd.FileName;
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.InitialDirectory = Application.StartupPath;
            ofd.Filter = "图片|*.jpg;*.jpeg;*.png";
            ofd.RestoreDirectory = true;
            var show = ofd.ShowDialog();
            if (show == DialogResult.OK)
            {
                textBox2.Text = ofd.FileName;
            }
        }

        private void button3_Click(object sender, EventArgs e)
        {
            FolderBrowserDialog folderBrowserDialog = new FolderBrowserDialog();
            //设置打开的标题Description    这里与文件不相同，文件是Title
            folderBrowserDialog.Description = "文件夹浏览";
            //判断打开的窗体中是否点了确定
            if (folderBrowserDialog.ShowDialog() == DialogResult.OK)
            {
                //获取到文件夹的路径  选中文件夹的路径SelectedPath
                string path = folderBrowserDialog.SelectedPath;
                textBox3.Text = path;
            }
        }

        private void button4_Click(object sender, EventArgs e)
        {
            FolderBrowserDialog folderBrowserDialog = new FolderBrowserDialog();
            //设置打开的标题Description    这里与文件不相同，文件是Title
            folderBrowserDialog.Description = "文件夹浏览";
            //判断打开的窗体中是否点了确定
            if (folderBrowserDialog.ShowDialog() == DialogResult.OK)
            {
                //获取到文件夹的路径  选中文件夹的路径SelectedPath
                string path = folderBrowserDialog.SelectedPath;
                textBox4.Text = path;
            }
        }

        private void button5_Click(object sender, EventArgs e)
        {
            if (string.IsNullOrEmpty(textBox4.Text))
            {
                MessageBox.Show("保存结果路径不能为空");
                return;
            }
            if (string.IsNullOrEmpty(textBox2.Text))
            {
                MessageBox.Show("图片不能为空");
                return;
            }
            Save(textBox2.Text);
            MessageBox.Show("保存成功");
        }

        private void button6_Click(object sender, EventArgs e)
        { 
            if (string.IsNullOrEmpty(textBox4.Text))
            {
                MessageBox.Show("保存结果路径不能为空");
                return;
            }
            if (string.IsNullOrEmpty(textBox3.Text))
            {
                MessageBox.Show("图片路径不能为空");
                return;
            }
            List<FileInfo> folder = new DirectoryInfo(textBox3.Text).GetFiles().ToList();
            foreach (FileInfo file in folder)
            {
                Save(file.FullName);
            }
            MessageBox.Show("保存成功");
        }

        private void Save(string file)
        {
            using var image = Image.FromFile(file);
            var labels = textBox5.Text.Split(",").ToList();
            using var scorer = new YoloScorerV2<YoloCocoP6Model>(textBox1.Text,labels);
            List<YoloPrediction> predictions = scorer.Predict(image);
            using var graphics = Graphics.FromImage(image);
            foreach (var prediction in predictions) // iterate predictions to draw results
            {
                double score = Math.Round(prediction.Score, 2);
                graphics.DrawRectangles(new Pen(prediction.Label.Color, 1),
                    new[] { prediction.Rectangle });
                var (x, y) = (prediction.Rectangle.X - 3, prediction.Rectangle.Y - 23);
                graphics.DrawString($"{prediction.Label.Name} ({score})",
                    new Font("Arial", 16, GraphicsUnit.Pixel), new SolidBrush(prediction.Label.Color),
                    new PointF(x, y));
            }
            image.Save(Path.Combine(textBox4.Text, $"{DateTime.Now.ToString("yyyyMMddHHmmssfff")}.jpg"));
        }

    }
}
