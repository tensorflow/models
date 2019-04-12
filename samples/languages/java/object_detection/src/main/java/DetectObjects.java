/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import static object_detection.protos.StringIntLabelMapOuterClass.StringIntLabelMap;
import static object_detection.protos.StringIntLabelMapOuterClass.StringIntLabelMapItem;

import com.google.protobuf.TextFormat;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import javax.imageio.ImageIO;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.tensorflow.framework.MetaGraphDef;
import org.tensorflow.framework.SignatureDef;
import org.tensorflow.framework.TensorInfo;
import org.tensorflow.types.UInt8;

/**
 * Java inference for the Object Detection API at:
 * https://github.com/tensorflow/models/blob/master/research/object_detection/
 */
public class DetectObjects {
  public static void main(String[] args) throws Exception {
    if (args.length < 3) {
      printUsage(System.err);
      System.exit(1);
    }
    final String[] labels = loadLabels(args[1]);
    try (SavedModelBundle model = SavedModelBundle.load(args[0], "serve")) {
      printSignature(model);
      for (int arg = 2; arg < args.length; arg++) {
        final String filename = args[arg];
        List<Tensor<?>> outputs = null;
        try (Tensor<UInt8> input = makeImageTensor(filename)) {
          outputs =
              model
                  .session()
                  .runner()
                  .feed("image_tensor", input)
                  .fetch("detection_scores")
                  .fetch("detection_classes")
                  .fetch("detection_boxes")
                  .run();
        }
        try (Tensor<Float> scoresT = outputs.get(0).expect(Float.class);
            Tensor<Float> classesT = outputs.get(1).expect(Float.class);
            Tensor<Float> boxesT = outputs.get(2).expect(Float.class)) {
          // All these tensors have:
          // - 1 as the first dimension
          // - maxObjects as the second dimension
          // While boxesT will have 4 as the third dimension (2 sets of (x, y) coordinates).
          // This can be verified by looking at scoresT.shape() etc.
          int maxObjects = (int) scoresT.shape()[1];
          float[] scores = scoresT.copyTo(new float[1][maxObjects])[0];
          float[] classes = classesT.copyTo(new float[1][maxObjects])[0];
          float[][] boxes = boxesT.copyTo(new float[1][maxObjects][4])[0];
          // Print all objects whose score is at least 0.5.
          System.out.printf("* %s\n", filename);
          boolean foundSomething = false;
          for (int i = 0; i < scores.length; ++i) {
            if (scores[i] < 0.5) {
              continue;
            }
            foundSomething = true;
            System.out.printf("\tFound %-20s (score: %.4f)\n", labels[(int) classes[i]], scores[i]);
          }
          if (!foundSomething) {
            System.out.println("No objects detected with a high enough score.");
          }
        }
      }
    }
  }

  private static void printSignature(SavedModelBundle model) throws Exception {
    MetaGraphDef m = MetaGraphDef.parseFrom(model.metaGraphDef());
    SignatureDef sig = m.getSignatureDefOrThrow("serving_default");
    int numInputs = sig.getInputsCount();
    int i = 1;
    System.out.println("MODEL SIGNATURE");
    System.out.println("Inputs:");
    for (Map.Entry<String, TensorInfo> entry : sig.getInputsMap().entrySet()) {
      TensorInfo t = entry.getValue();
      System.out.printf(
          "%d of %d: %-20s (Node name in graph: %-20s, type: %s)\n",
          i++, numInputs, entry.getKey(), t.getName(), t.getDtype());
    }
    int numOutputs = sig.getOutputsCount();
    i = 1;
    System.out.println("Outputs:");
    for (Map.Entry<String, TensorInfo> entry : sig.getOutputsMap().entrySet()) {
      TensorInfo t = entry.getValue();
      System.out.printf(
          "%d of %d: %-20s (Node name in graph: %-20s, type: %s)\n",
          i++, numOutputs, entry.getKey(), t.getName(), t.getDtype());
    }
    System.out.println("-----------------------------------------------");
  }

  private static String[] loadLabels(String filename) throws Exception {
    String text = new String(Files.readAllBytes(Paths.get(filename)), StandardCharsets.UTF_8);
    StringIntLabelMap.Builder builder = StringIntLabelMap.newBuilder();
    TextFormat.merge(text, builder);
    StringIntLabelMap proto = builder.build();
    int maxId = 0;
    for (StringIntLabelMapItem item : proto.getItemList()) {
      if (item.getId() > maxId) {
        maxId = item.getId();
      }
    }
    String[] ret = new String[maxId + 1];
    for (StringIntLabelMapItem item : proto.getItemList()) {
      ret[item.getId()] = item.getDisplayName();
    }
    return ret;
  }

  private static void bgr2rgb(byte[] data) {
    for (int i = 0; i < data.length; i += 3) {
      byte tmp = data[i];
      data[i] = data[i + 2];
      data[i + 2] = tmp;
    }
  }

  private static Tensor<UInt8> makeImageTensor(String filename) throws IOException {
    BufferedImage img = ImageIO.read(new File(filename));
    if (img.getType() != BufferedImage.TYPE_3BYTE_BGR) {
      BufferedImage newImage = new BufferedImage(
          img.getWidth(), img.getHeight(), BufferedImage.TYPE_3BYTE_BGR);
      Graphics2D g = newImage.createGraphics();
      g.drawImage(img, 0, 0, img.getWidth(), img.getHeight(), null);
      g.dispose();
      img = newImage;
    }
      
    byte[] data = ((DataBufferByte) img.getData().getDataBuffer()).getData();
    // ImageIO.read seems to produce BGR-encoded images, but the model expects RGB.
    bgr2rgb(data);
    final long BATCH_SIZE = 1;
    final long CHANNELS = 3;
    long[] shape = new long[] {BATCH_SIZE, img.getHeight(), img.getWidth(), CHANNELS};
    return Tensor.create(UInt8.class, shape, ByteBuffer.wrap(data));
  }

  private static void printUsage(PrintStream s) {
    s.println("USAGE: <model> <label_map> <image> [<image>] [<image>]");
    s.println("");
    s.println("Where");
    s.println("<model> is the path to the SavedModel directory of the model to use.");
    s.println("        For example, the saved_model directory in tarballs from ");
    s.println(
        "        https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)");
    s.println("");
    s.println(
        "<label_map> is the path to a file containing information about the labels detected by the model.");
    s.println("            For example, one of the .pbtxt files from ");
    s.println(
        "            https://github.com/tensorflow/models/tree/master/research/object_detection/data");
    s.println("");
    s.println("<image> is the path to an image file.");
    s.println("        Sample images can be found from the COCO, Kitti, or Open Images dataset.");
    s.println(
        "        See: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md");
  }
}
