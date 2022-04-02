using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace weights2caffe {
    class tool {
        public static bool data(string dst, string modelName, 
                                uint n, uint c, uint h, uint w) {

            StreamWriter write = new StreamWriter(dst + ".prototxt", false);
            write.WriteLine("name: " + "\"" + modelName + "\"");
            write.WriteLine("input: " + "\"data\"");
            write.WriteLine("input_shape {");
            write.WriteLine("  dim: " + n);
            write.WriteLine("  dim: " + c);
            write.WriteLine("  dim: " + h);
            write.WriteLine("  dim: " + w);
            write.WriteLine("}");
            write.Close();

            write = new StreamWriter(dst + ".txt", false);
            write.WriteLine("name: " + "\"" + modelName + "\"");
            write.WriteLine("layer {");
            write.WriteLine("  name: " + "\"input\"");
            write.WriteLine("  top: " + "\"data\"");
            write.WriteLine("  type: " + "\"Input\"");
            write.WriteLine("  phase: TEST");
            write.WriteLine("  input_param {");
            write.WriteLine("    shape {");
            write.WriteLine("      dim: " + n);
            write.WriteLine("      dim: " + c);
            write.WriteLine("      dim: " + h);
            write.WriteLine("      dim: " + w);
            write.WriteLine("    }");
            write.WriteLine("  }");
            write.WriteLine("}");
            write.Close();

            return true;
        }
        public static bool conv2d(string dst, string src, 
                                  string top, string bottom, 
                                  bool bias, 
                                  uint stride, uint pad_h, uint pad_w,
                                  uint n, uint c, uint h, uint w) {

            uint size = n * c * h * w;
            StreamWriter write = new StreamWriter(dst + ".prototxt", true);
            write.WriteLine("layer {");
            write.WriteLine("  name: \"" + top + "\"");
            write.WriteLine("  top: \"" + top + "\"");
            write.WriteLine("  bottom: \"" + bottom + "\"");
            write.WriteLine("  type: \"Convolution\"");
            write.WriteLine("  convolution_param {");
            write.WriteLine("    num_output: " + n);
            write.WriteLine("    kernel_h: " + h);
            write.WriteLine("    kernel_w: " + w);
            write.WriteLine("    stride: " + stride);
            write.WriteLine("    pad_h: " + pad_h);
            write.WriteLine("    pad_w: " + pad_w);
            write.WriteLine("    bias_term: " + bias.ToString().ToLower());
            write.WriteLine("  }");
            write.WriteLine("}");
            write.Close();

            string[] tmp = File.ReadAllLines(src);
            write = new StreamWriter(dst + ".txt", true);
            write.WriteLine("layer {");
            write.WriteLine("  name: \"" + top + "\"");
            write.WriteLine("  top: \"" + top + "\"");
            write.WriteLine("  bottom: \"" + bottom + "\"");
            write.WriteLine("  type: \"Convolution\"");
            write.WriteLine("  blobs {");
            for (uint i = 0; i < size; i++) {
                write.WriteLine("    data: " + float.Parse(tmp[i]));
            }
            write.WriteLine("    shape {");
            write.WriteLine("      dim: " + n);
            write.WriteLine("      dim: " + c);
            write.WriteLine("      dim: " + h);
            write.WriteLine("      dim: " + w);
            write.WriteLine("    }");
            write.WriteLine("  }");
            if (bias) {
                write.WriteLine("  blobs {");
                for (uint i = 0; i < n; i++) {
                    write.WriteLine("    data: " + float.Parse(tmp[size + i]));
                }
                write.WriteLine("    shape {");
                write.WriteLine("      dim: " + n);
                write.WriteLine("    }");
                write.WriteLine("  }");
            }
            write.WriteLine("  phase: TEST");
            write.WriteLine("  convolution_param {");
            write.WriteLine("    num_output: " + n);
            write.WriteLine("    kernel_h: " + h);
            write.WriteLine("    kernel_w: " + w);
            write.WriteLine("    stride: " + stride);
            write.WriteLine("    pad_h: " + pad_h);
            write.WriteLine("    pad_w: " + pad_w);
            write.WriteLine("    bias_term: " + bias.ToString().ToLower());
            write.WriteLine("  }");
            write.WriteLine("}");
            write.Close();

            return true;
        }
        static bool relu(string dst, 
                         string top, 
                         float relu_param) {

            StreamWriter write = new StreamWriter(dst + ".prototxt", true);
            write.WriteLine("layer {");
            write.WriteLine("  name: \"" + top + "-act\"");
            write.WriteLine("  top: \"" + top + "\"");
            write.WriteLine("  bottom: \"" + top + "\"");
            write.WriteLine("  type: \"ReLU\"");
            write.WriteLine("  relu_param {");
            write.WriteLine("    negative_slope: " + relu_param);
            write.WriteLine("  }");
            write.WriteLine("}");
            write.Close();

            write = new StreamWriter(dst + ".txt", true);
            write.WriteLine("layer {");
            write.WriteLine("  name: \"" + top + "-act\"");
            write.WriteLine("  top: \"" + top + "\"");
            write.WriteLine("  bottom: \"" + top + "\"");
            write.WriteLine("  type: \"ReLU\"");
            write.WriteLine("  phase: TEST");
            write.WriteLine("  relu_param {");
            write.WriteLine("    negative_slope: " + relu_param);
            write.WriteLine("  }");
            write.WriteLine("}");
            write.Close();

            return true;
        }
        static bool bn(string dst, string src,
                       string top,
                       uint n) {

            StreamWriter write = new StreamWriter(dst + ".prototxt", true);
            write.WriteLine("layer {");
            write.WriteLine("  name: \"" + top + "-bn\"");
            write.WriteLine("  top: \"" + top + "\"");
            write.WriteLine("  bottom: \"" + top + "\"");
            write.WriteLine("  type: \"BatchNorm\"");
            write.WriteLine("  batch_norm_param {");
            write.WriteLine("    use_global_stats: true");
            write.WriteLine("  }");
            write.WriteLine("}");
            write.Close();

            string[] tmp = File.ReadAllLines(src);
            write = new StreamWriter(dst + ".txt", true);
            write.WriteLine("layer {");
            write.WriteLine("  name: \"" + top + "-bn\"");
            write.WriteLine("  top: \"" + top + "\"");
            write.WriteLine("  bottom: \"" + top + "\"");
            write.WriteLine("  type: \"BatchNorm\"");
            for (int i = 0; i < 3; i++) {
                write.WriteLine("  param {");
                write.WriteLine("    lr_mult: 0");
                write.WriteLine("  }");
            }
            write.WriteLine("  blobs {");
            for (uint i = 0; i < n; i++) {
                write.WriteLine("    data: " + float.Parse(tmp[i].Split(',')[2]));
            }
            write.WriteLine("    shape {");
            write.WriteLine("      dim: " + n);
            write.WriteLine("    }");
            write.WriteLine("  }");
            write.WriteLine("  blobs {");
            for (uint i = 0; i < n; i++) {
                write.WriteLine("    data: " + float.Parse(tmp[i].Split(',')[3]));
            }
            write.WriteLine("    shape {");
            write.WriteLine("      dim: " + n);
            write.WriteLine("    }");
            write.WriteLine("  }");
            write.WriteLine("  blobs {");
            write.WriteLine("    data: 1");
            write.WriteLine("    shape {");
            write.WriteLine("      dim: 1");
            write.WriteLine("    }");
            write.WriteLine("  }");
            write.WriteLine("  phase: TEST");
            write.WriteLine("  batch_norm_param {");
            write.WriteLine("    use_global_stats: true");
            write.WriteLine("  }");
            write.WriteLine("}");
            write.Close();

            return true;
        }
        static bool scale(string dst, string src,
                          string top,
                          uint n) {

            StreamWriter write = new StreamWriter(dst + ".prototxt", true);
            write.WriteLine("layer {");
            write.WriteLine("  name: \"" + top + "-scale\"");
            write.WriteLine("  top: \"" + top + "\"");
            write.WriteLine("  bottom: \"" + top + "\"");
            write.WriteLine("  type: \"Scale\"");
            write.WriteLine("  scale_param {");
            write.WriteLine("    bias_term: true");
            write.WriteLine("  }");
            write.WriteLine("}");
            write.Close();

            string[] tmp = File.ReadAllLines(src);
            write = new StreamWriter(dst + ".txt", true);
            write.WriteLine("layer {");
            write.WriteLine("  name: \"" + top + "-scale\"");
            write.WriteLine("  top: \"" + top + "\"");
            write.WriteLine("  bottom: \"" + top + "\"");
            write.WriteLine("  type: \"Scale\"");
            write.WriteLine("  blobs {");
            for (uint i = 0; i < n; i++) {
                write.WriteLine("    data: " + float.Parse(tmp[i].Split(',')[0]));
            }
            write.WriteLine("    shape {");
            write.WriteLine("      dim: " + n);
            write.WriteLine("    }");
            write.WriteLine("  }");
            write.WriteLine("  blobs {");
            for (uint i = 0; i < n; i++) {
                write.WriteLine("    data: " + float.Parse(tmp[i].Split(',')[1]));
            }
            write.WriteLine("    shape {");
            write.WriteLine("      dim: " + n);
            write.WriteLine("    }");
            write.WriteLine("  }");
            write.WriteLine("  phase: TEST");
            write.WriteLine("  scale_param {");
            write.WriteLine("    bias_term: true");
            write.WriteLine("  }");
            write.WriteLine("}");
            write.Close();

            return true;
        }
        public static bool conv2d_relu(string dst, string src,
                                       string top, string bottom,
                                       bool bias,
                                       float relu_param,
                                       uint stride, uint pad_h, uint pad_w,
                                       uint n, uint c, uint h, uint w) {

            conv2d(dst, src + "_CONV_BIAS.csv",
                   top, bottom,
                   bias,
                   stride, pad_h, pad_w,
                   n, c, h, w);
            relu(dst,
                 top,
                 relu_param);

            return true;
        }
        public static bool bnconv2d_relu(string dst, string src,
                                         string top, string bottom,
                                         float relu_param,
                                         uint stride, uint pad_h, uint pad_w,
                                         uint n, uint c, uint h, uint w) {

            conv2d(dst, src + "_CONV.csv",
                   top, bottom,
                   false,
                   stride, pad_h, pad_w,
                   n, c, h, w);

            bn(dst, src + "_CGBMV.csv",
               top,
               n);

            scale(dst, src + "_CGBMV.csv",
                  top,
                  n);

            relu(dst,
                 top,
                 relu_param);

            return true;
        }
        public static bool mp2d(string dst,
                                string top, string bottom, 
                                uint stride, 
                                uint pad,
                                uint h, uint w) {

            StreamWriter write = new StreamWriter(dst + ".prototxt", true);
            write.WriteLine("layer {");
            write.WriteLine("  name: \"" + top + "\"");
            write.WriteLine("  top: \"" + top + "\"");
            write.WriteLine("  bottom: \"" + bottom + "\"");
            write.WriteLine("  type: \"Pooling\"");
            write.WriteLine("  pooling_param {");
            write.WriteLine("    pool: MAX");
            write.WriteLine("    kernel_h: " + h);
            write.WriteLine("    kernel_w: " + w);
            write.WriteLine("    stride: " + stride);
            write.WriteLine("    pad: " + pad);
            write.WriteLine("  }");
            write.WriteLine("}");
            write.Close();

            write = new StreamWriter(dst + ".txt", true);
            write.WriteLine("layer {");
            write.WriteLine("  name: \"" + top + "\"");
            write.WriteLine("  top: \"" + top + "\"");
            write.WriteLine("  bottom: \"" + bottom + "\"");
            write.WriteLine("  type: \"Pooling\"");
            write.WriteLine("  phase: TEST");
            write.WriteLine("  pooling_param {");
            write.WriteLine("    pool: MAX");
            write.WriteLine("    kernel_h: " + h);
            write.WriteLine("    kernel_w: " + w);
            write.WriteLine("    stride: " + stride);
            write.WriteLine("    pad: " + pad);
            write.WriteLine("  }");
            write.WriteLine("}");
            write.Close();

            return true;
        }
        public static bool ups(string dst, 
                               string top, string bottom, 
                               uint scale) {

            StreamWriter write = new StreamWriter(dst + ".prototxt", true);
            write.WriteLine("layer {");
            write.WriteLine("  name: \"" + top + "\"");
            write.WriteLine("  top: \"" + top + "\"");
            write.WriteLine("  bottom: \"" + bottom + "\"");
            write.WriteLine("  type: \"Upsample\"");
            write.WriteLine("  upsample_param {");
            write.WriteLine("    scale: " + scale);
            write.WriteLine("  }");
            write.WriteLine("}");
            write.Close();

            write = new StreamWriter(dst + ".txt", true);
            write.WriteLine("layer {");
            write.WriteLine("  name: \"" + top + "\"");
            write.WriteLine("  top: \"" + top + "\"");
            write.WriteLine("  bottom: \"" + bottom + "\"");
            write.WriteLine("  type: \"Upsample\"");
            write.WriteLine("  phase: TEST");
            write.WriteLine("  upsample_param {");
            write.WriteLine("    scale: " + scale);
            write.WriteLine("  }");
            write.WriteLine("}");
            write.Close();

            return true;
        }
        public static bool concat(string dst,
                                  string top,
                                  string bottom1,
                                  string bottom2) {

            StreamWriter write = new StreamWriter(dst + ".prototxt", true);
            write.WriteLine("layer {");
            write.WriteLine("  name: \"" + top + "\"");
            write.WriteLine("  top: \"" + top + "\"");
            write.WriteLine("  bottom: \"" + bottom1 + "\"");
            write.WriteLine("  bottom: \"" + bottom2 + "\"");
            write.WriteLine("  type: \"Concat\"");
            write.WriteLine("}");
            write.Close();

            write = new StreamWriter(dst + ".txt", true);
            write.WriteLine("layer {");
            write.WriteLine("  name: \"" + top + "\"");
            write.WriteLine("  top: \"" + top + "\"");
            write.WriteLine("  bottom: \"" + bottom1 + "\"");
            write.WriteLine("  bottom: \"" + bottom2 + "\"");
            write.WriteLine("  type: \"Concat\"");
            write.WriteLine("  phase: TEST");
            write.WriteLine("}");
            write.Close();

            return true;
        }
        public static bool concat(string dst,
                                  string top,
                                  string bottom1,
                                  string bottom2,
                                  string bottom3) {

            StreamWriter write = new StreamWriter(dst + ".prototxt", true);
            write.WriteLine("layer {");
            write.WriteLine("  name: \"" + top + "\"");
            write.WriteLine("  top: \"" + top + "\"");
            write.WriteLine("  bottom: \"" + bottom1 + "\"");
            write.WriteLine("  bottom: \"" + bottom2 + "\"");
            write.WriteLine("  bottom: \"" + bottom3 + "\"");
            write.WriteLine("  type: \"Concat\"");
            write.WriteLine("}");
            write.Close();

            write = new StreamWriter(dst + ".txt", true);
            write.WriteLine("layer {");
            write.WriteLine("  name: \"" + top + "\"");
            write.WriteLine("  top: \"" + top + "\"");
            write.WriteLine("  bottom: \"" + bottom1 + "\"");
            write.WriteLine("  bottom: \"" + bottom2 + "\"");
            write.WriteLine("  bottom: \"" + bottom3 + "\"");
            write.WriteLine("  type: \"Concat\"");
            write.WriteLine("  phase: TEST");
            write.WriteLine("}");
            write.Close();

            return true;
        }
        public static bool concat(string dst,
                                  string top, 
                                  string bottom1,
                                  string bottom2, 
                                  string bottom3,
                                  string bottom4) {

            StreamWriter write = new StreamWriter(dst + ".prototxt", true);
            write.WriteLine("layer {");
            write.WriteLine("  name: \"" + top + "\"");
            write.WriteLine("  top: \"" + top + "\"");
            write.WriteLine("  bottom: \"" + bottom1 + "\"");
            write.WriteLine("  bottom: \"" + bottom2 + "\"");
            write.WriteLine("  bottom: \"" + bottom3 + "\"");
            write.WriteLine("  bottom: \"" + bottom4 + "\"");
            write.WriteLine("  type: \"Concat\"");
            write.WriteLine("}");
            write.Close();

            write = new StreamWriter(dst + ".txt", true);
            write.WriteLine("layer {");
            write.WriteLine("  name: \"" + top + "\"");
            write.WriteLine("  top: \"" + top + "\"");
            write.WriteLine("  bottom: \"" + bottom1 + "\"");
            write.WriteLine("  bottom: \"" + bottom2 + "\"");
            write.WriteLine("  bottom: \"" + bottom3 + "\"");
            write.WriteLine("  bottom: \"" + bottom4 + "\"");
            write.WriteLine("  type: \"Concat\"");
            write.WriteLine("  phase: TEST");
            write.WriteLine("}");
            write.Close();

            return true;
        }
        public static bool eltwise(string dst,
                                   string top, 
                                   string bottom1,
                                   string bottom2) {

            StreamWriter write = new StreamWriter(dst + ".prototxt", true);
            write.WriteLine("layer {");
            write.WriteLine("  name: \"" + top + "\"");
            write.WriteLine("  top: \"" + top + "\"");
            write.WriteLine("  bottom: \"" + bottom1 + "\"");
            write.WriteLine("  bottom: \"" + bottom2 + "\"");
            write.WriteLine("  type: \"Eltwise\"");
            write.WriteLine("  eltwise_param {");
            write.WriteLine("    operation: SUM");
            write.WriteLine("  }");
            write.WriteLine("}");
            write.Close();

            write = new StreamWriter(dst + ".txt", true);
            write.WriteLine("layer {");
            write.WriteLine("  name: \"" + top + "\"");
            write.WriteLine("  top: \"" + top + "\"");
            write.WriteLine("  bottom: \"" + bottom1 + "\"");
            write.WriteLine("  bottom: \"" + bottom2 + "\"");
            write.WriteLine("  type: \"Eltwise\"");
            write.WriteLine("  phase: TEST");
            write.WriteLine("  eltwise_param {");
            write.WriteLine("    operation: SUM");
            write.WriteLine("  }");
            write.WriteLine("}");
            write.Close();

            return true;
        }

    }
}
