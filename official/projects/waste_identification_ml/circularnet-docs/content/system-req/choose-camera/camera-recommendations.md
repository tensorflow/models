The following list contains the essential recommendations when selecting a
camera to capture high-quality images:

-  **Frame rate:** Ensure the camera can capture images at a high frame rate,
   matching the speed of the conveyor belt to avoid motion blurriness. Look for
   cameras with at least 30 FPS (frames per second) or higher.
-  **Shutter type:** A global shutter is preferable over a rolling shutter to
   avoid distortion, especially for fast-moving objects.
-  **Lens compatibility:** Choose a camera with interchangeable lenses to adjust
   the field of view and focus based on the conveyor belt width and object size.
-  **Interface:** Select a camera with an appropriate interface (for example,
   USB 3.0, GigE, or Camera Link) that supports high-speed data transfer to the
   processing unit.
-  **Integration with software libraries:** Verify compatibility with software
   and libraries for seamless integration. For example, if you are using an edge
   device, such as NVIDIA or Raspberry Pi, you need software integration
   compatibility with it.
-  **Scan type:** An area scan camera is preferable over a line scan camera to
   ensure flexibility and comprehensive imaging capabilities required to
   identify varying waste materials effectively. Area scan cameras are suitable
   when objects vary significantly in size and shape and for situations where
   the conveyor belt speed varies or is not uniform.
-  **Resolution:** A suitable resolution is crucial for image analysis because
   it corresponds to the level of detail, patterns, and textures a camera can
   detect from an object. You must have an image resolution of at least
   1024x1024 pixels. For this reason, choose a camera with a resolution between
   0.5 and 1.0 MP (megapixels), depending on the size of the objects and the
   level of detail required.

The higher the pixel resolution, the more processing time and latency are
introduced by having to process all those pixels through the model. Find a
balance between image resolution and processing latency.

The following list contains additional factors you might want to consider based
on your facility's conditions:

-  **Enclosure:** Ensure the camera has an industrial-grade enclosure,
   preferably NEMA, IP65, or higher, to withstand harsh environments and dust.
   For example, [Basler has some offerings](https://www.baslerweb.com/en/products/accessories-and-bundles/basler-ip67-housing/).
-  **Lighting:** Ensure diffused or even lighting across all items on the
   conveyor belt. If the conveyor belt's speed is high, you need a smaller
   [aperture size](./factors/#aperture-size-f-number) and a higher [shutter
   speed](./factors/#shutter-speed). The conveyor belt should be well-lit to
   ensure bright images, reducing blurriness.
-  **Parameter control:** Opt for a camera with good low-light sensitivity and
   high dynamic range to handle varying lighting conditions. Look for cameras
   with adjustable exposure settings to accommodate different lighting
   conditions and object speeds. Also, choose a camera that supports remote
   configuration for tuning software parameters such as shutter speed, exposure,
   and frame rate.
-  **Integrated lightning:** Accompany the camera with an integrated lightning
   system to reach the required luminosity in your facility. You can find
   accessories such as lightning bulbs or lamps to increase the brightness of
   the conveyor belt.
-  **Synchronization:** If you need multiple cameras, ensure they can be
   synchronized to capture images simultaneously.
-  **Power supply:** Consider power over ethernet (PoE).
-  **Temperature range:** Verify the camera can operate within the temperature
   range of the recycling facility.
-  **Reliability and durability:** Choose cameras with a proven track record for
   reliability and durability in industrial applications.

### **Camera Installation and Placement**

Proper installation involves positioning the camera **directly** **above** the
conveyor belt for **consistent** **lighting** and **full** **coverage**. As a
general rule, avoid capturing images with motion blur, fisheye effect,
and quality issues due to vibrations from the conveyor belt movement;
global shutter cameras typically meet these recommendations. Crucially, no
hands or other foreign objects, **nor overflowing belt edges** should be
visible in the camera's field of view during operation, as
obstructions will interfere with object detection and classification.

---

#### **Camera Mounting Guidelines:**

Camera mounting directly impacts image stability and clarity. Adhering to
these guidelines is critical for reliable data acquisition and consistent
model performance.

* **Vertical and Centered Alignment**: The camera must be positioned directly
    and vertically above (i.e. perpendicular) the exact center of the
    conveyor belt. This ensures a consistent perspective, minimizes image
    distortion from angled views, and provides uniform coverage of the
    belt's width.
* **Fixed and Optimized Height**: The camera must be mounted at a fixed,
    immovable height above the conveyor belt. This height should be
    precisely determined based on calculations for optimal focal length
    and field of view (FoV), considering the conveyor belt's dimensions
    (width and object height). Once established, this height must remain
    constant to maintain consistent image scaling and object detection
    accuracy. Typical camera heights in facilities are around 3-4 feet
    (approximately 1-2 meters). A precise 1:1 ratio between the camera's
    working distance (height) and the field of view (e.g., a 1-meter
    height for a 1x1 meter FoV) is highly recommended for optimal image
    proportionality and model performance. Smaller belt widths and lower
    camera heights are generally preferred where feasible to achieve
    higher pixel density per object.
* **Robust and Vibration-Resistant Assembly**: Utilize industrial-grade
    mounting hardware specifically designed for high-vibration
    environments like MRFs. The mounting system must prevent any
    perceptible movement, wobble, or vibration of the camera, which would
    result in blurry images and severely impact image quality and model
    performance. Regular inspections for mount integrity and tightness are
    mandatory.
* **Unobstructed Field of View**: The mounting hardware itself, all
    associated cabling, and any other facility infrastructure must be
    positioned entirely outside the camera's field of view. A clear and
    unobstructed view of the materials on the conveyor belt is essential
    for accurate object detection and classification.
* **Accessibility for Maintenance (Secondary Consideration)**: While the
    primary focus is on stability and an unobstructed view, consider
    designing the mounting setup to allow for reasonable access for
    routine maintenance, cleaning of the camera lens, and adjustments to
    lighting components.

---

#### **Strict Lighting Guidelines:**

Consistent and adequate lighting is paramount for optimal image quality and
accurate material identification.

* **Diffused and Even Illumination**: Ensure [diffused or even lighting](https://www.effilux.com/en/products/led-bar/effi-flex)
    across all items on the conveyor belt. Diffused lighting refers to
    light that has been spread out or softened, rather than being direct
    and harsh. This can be achieved using [diffusers](https://www.amazon.com/Torjim-Photography-Professional-3000-7500K-Recording/dp/B0CF44WSPJ/ref=sr_1_6?dib=eyJ2IjoiMSJ9.t7O2HZqH9szXiU7jZ2GIdULPAq9kN2Wqwo5ESFO5NDPH47xTxiugxvEh1lnsvCbd38rzWAZnNci8eiJfJtdzL-FDVJT3uZAzdvVz8QcqUiZA96QcZ2YmoUxUFLrTuOYi9VL7GJM6nrc1gjbAyR6M__NuvtTgtJ8WJKVvDiMubuEfBM7OEkHWT_3tw00_bNHTvB95rotyGse14vDsH9O7KDnDJggn5fIgW09tyNDf4dc.f9-abHQLKPtm52XxLyl0oXmSzGHUqRKZMO53S76LdhQ&dib_tag=se&keywords=light%2Bdiffuser&qid=1750113487&sr=8-6&th=1)
    (translucent materials placed between the light source and the subject)
    or by using large, soft light sources. The goal is to minimize harsh
    shadows and hot spots, which can obscure material features and
    negatively impact model performance.
* **High-Speed Belt Lighting for Motion Blur Prevention**: If the conveyor
    belt's speed is high, a smaller aperture size and a higher shutter
    speed are necessary to prevent motion blur. To compensate for the
    reduced light intake at higher shutter speeds, the conveyor belt
    must be exceptionally well-lit to ensure bright images.
* **Integrated and Dedicated Lighting System**: Accompany the camera with a
    dedicated integrated lighting system to achieve the required
    luminosity in your facility. This system should provide ample
    brightness for the entire field of view.
* **Eliminate Lighting Fluctuations**: Minimize any lighting fluctuations
    (e.g., flickering lights, inconsistent ambient light, direct and even
    indirect sunlight exposure) that could introduce variability into the
    images and hinder consistent object detection. The lighting
    environment should be controlled and stable throughout operation.
* **Color Temperature Consistency**: Maintain a consistent color temperature
    for all light sources. Variations in color temperature can alter the
    perceived color of materials, potentially impacting the model's
    ability to accurately classify objects, especially when color is a
    distinguishing feature.
* **Adequate Lux/Luminosity**: The lighting system must provide sufficient
    lux (lumens per square meter) to ensure that the camera sensor
    receives enough light for clear image capture, even at higher shutter
    speeds. This is crucial for optimal image quality and model
    performance.

*Diffused lighting helps capture the full spectrum of light for an object
and minimizes shadows*

---

#### **Factors Based on Vision System Placement:**

You must precisely measure and account for the following factors when
installing the camera above the conveyor belt:

* **Conveyor Belt Width**: Accurately determine the total operational width
    of the conveyor belt that the camera needs to fully cover. This
    measurement is crucial for selecting the appropriate lens focal length
    and ensuring the camera's field of view encompasses the entire belt
    width *without* extending beyond its edges. Typical belt widths range
    between 1 and 1.5 meters.
* **Camera Height (Critical for Focal Length and FoV Ratio)**: Confirm the
    fixed vertical height above the conveyor belt at which the camera
    will be mounted. This height directly influences the required focal
    length. Standard camera heights in facilities are around 3-4 feet
    (approximately 1-2 meters). A precise 1:1 ratio between the camera's
    working distance (height) and the field of view (e.g., a 1-meter
    height for a 1x1 meter FoV) is highly recommended for optimal image
    proportionality and model performance. For example, if your FoV
    captures an area of 1x1 square meters of the belt, you must mount the
    camera one or 1.5 meters above the conveyor belt.
* **Conveyor Belt Speed (Critical for Shutter Speed)**: Accurately measure
    the average and maximum operational speed of the conveyor belt. This
    speed is a primary determinant for calculating the adequate shutter
    speed to minimize motion blur. Belt speeds typically range between 1
    and 4 meters per second; slower speeds are always preferable for
    achieving sharper images.
* **Field of View (FoV) and Image Proportionality (Square Aspect Ratio)**:
    While the conveyor belt length is continuous, the camera's FoV must
    capture well-proportioned images. The ideal FoV should be
    approximately square, meaning the length of the belt captured in the
    frame should be similar to its width (e.g., a 1-meter belt width
    should correspond to approximately 1 meter of belt length in the
    image). This square aspect ratio assists in consistent object
    detection and tracking.
* **Minimize Item Overlap**: The camera's positioning, coupled with
    optimized conveyor belt loading procedures, should actively aim to
    minimize the overlap of items on the belt. Excessive overlap
    significantly impedes accurate pixel-level instance segmentation and
    reduces detection confidence.
* **Consistent Object Size**: While not always controllable, the system is
    optimized for objects similar in size to those typically found in
    household recycling streams, as depicted in the provided visual
    examples.

---

#### **Camera Setup Checklist for CircularNet Deployment:**

**I. Camera Mounting & Placement:**

-   [ ] **Vertical Alignment:** Is the camera positioned directly and
        vertically above the exact center of the conveyor belt?
-   [ ] **Fixed Height:** Is the camera mounted at a fixed, immovable
        height determined by focal length and FoV calculations?
-   [ ] **Robust Mount:** Is industrial-grade, vibration-resistant
        mounting hardware being used for a stable assembly?
-   [ ] **Unobstructed FoV:** Are all mounting hardware, cables, and
        facility structures outside the camera's field of view?
-   [ ] **Belt Coverage:** Does the camera's FoV fully cover the entire
        width of the conveyor belt without extending beyond its edges?
-   [ ] **Square FoV:** Is the FoV configured to be approximately square
        (belt width:belt length ratio of ~1:1)?
-   [ ] **No Hands/Obstructions:** Are operational procedures in place to
        ensure no hands, foreign objects, or overflowing belt edges are
        visible in the camera's view?
-   [ ] **Minimize Overlap:** Are conveyor belt loading procedures
        optimized to minimize item overlap for better detection?

**II. Lighting System:**

-   [ ] **Diffused Illumination:** Is the lighting system designed to
        provide diffused and even illumination across the entire conveyor
        belt?
-   [ ] **Adequate Brightness:** Is the lighting system powerful enough to
        ensure bright images, especially at high conveyor belt speeds and
        faster shutter speeds?
-   [ ] **Dedicated Lighting:** Is an integrated and dedicated lighting
        system being installed with the camera?
-   [ ] **Stable Lighting:** Are measures in place to eliminate lighting
        fluctuations (flicker, ambient light changes)?
-   [ ] **Consistent Color Temperature:** Is the lighting system
        maintaining a consistent color temperature?
-   [ ] **Adequate Lux:** Is the lighting system providing sufficient
        lux/luminosity for clear image capture?

**III. Environmental Measurements:**

-   [ ] **Conveyor Belt Width:** Has the precise operational width of the
        conveyor belt been measured and are there no overflowing
        portions visible in sample images?
-   [ ] **Camera Mounting Height:** Has the exact fixed height for camera
        mounting above the belt been determined?
-   [ ] **Conveyor Belt Speed:** Has the average and maximum operational
        speed of the conveyor belt been measured?

---
## Recommended models

The following list contains some examples of recommended models for your camera:

-  [Arducam High Quality Camera](https://www.arducam.com/product/b0242-arducam-imx477-hq-camera/)
-  [GoPro HERO12 Black](https://gopro.com/en/us/shop/cameras/hero12-black/CHDHX-121-master.html)
