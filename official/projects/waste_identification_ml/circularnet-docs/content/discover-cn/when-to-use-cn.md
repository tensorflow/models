Consider CircularNet if you want to automate the analysis of waste composition
and material identification within your Material Recovery Facility (MRF) or
recycling center. It is particularly valuable for scenarios where you want to
implement the following functionalities:

-  **Gain aggregate or real-time insights** into the material types and forms
   moving through your facility on conveyor belts.
-  **Reduce reliance on manual inspection** and sorting, thereby improving
   efficiency and minimizing human error.
-  **Identify and quantify contaminants** within waste streams to improve the
   quality of recycled materials.
-  **Generate automated and historical reports** on material composition,
   recycling rates, and contamination levels to support data-driven
   decision-making and operational improvements.

CircularNet utilizes RGB computer vision models and pixel-level instance
segmentation to accurately identify and classify materials, making it a valuable
tool for enhancing the efficiency and effectiveness of waste management
operations.

CircularNet identifies material forms and types. Furthermore, in the case of
plastic, it identifies plastic types. The model employs pixel-level instance
segmentation, a technique that precisely outlines the shape of each object
within an image. This technique offers several advantages, such as the
following, compared to traditional bounding box object detection methods:

-  **Accurate object delineation**: Pixel-level segmentation provides a precise
   representation of object boundaries, which is critical for accurately
   measuring object size, shape, and quantity, especially in cluttered waste
   streams.
-  **Improved contamination detection**: By accurately segmenting objects,
   CircularNet can identify and quantify contaminants mixed with recyclables,
   leading to enhanced sorting and higher-quality recycled materials.
-  **Enhanced material characterization**: Pixel-level information enables
   nuanced material analysis, letting CircularNet distinguish between
   similar-looking materials or identify specific material attributes, such as
   plastic types.