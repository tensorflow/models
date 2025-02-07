You must measure the following factors when installing the camera above the
conveyor belt:

-  **Conveyor belt width:** Determine the total width of the conveyor belt that
   the camera needs to cover. This value helps determine the [focal length](#focal-length).
   The conveyor belt width is typically between one and
   1.5 meters, and the camera must capture the entire belt width on the image
   frame.
-  **Camera height:** Confirm the fixed height above the conveyor belt at which
   you will mount the camera. This value helps determine the [focal length](#focal-length).
   The camera height is typically between one and two
   meters above the conveyor belt.
-  **Conveyor belt speed:** Measure the conveyor belt average speed to determine
   the camera's adequate [shutter speed](#shutter-speed). The conveyor belt
   speed must be between one and four meters per second.

The conveyor belt length is variable because the belt moves continuously.
However, consider capturing well-proportioned images. The field of view (FoV) is
the area the camera covers on the captured images. This area must be
approximately a square, covering the entire belt width and a similar distance
for the belt length. So, for example, if the belt width is one meter, the belt
length captured in the frame should also be approximately one meter to cover a
square area.

Keep an approximate ratio of 1:1 between the FoV and the camera height. So, for
example, if your FoV captures an area of 1x1 square meters of the belt, you must
mount the camera one or 1.5 meters above the conveyor belt.

Calculate the following camera specifications based on the factors you measured:

{{< table_of_contents >}}

---

## Sensor size

Decide on the camera sensor size according to the information you need to
detect. Larger sensor sizes fit more information, while smaller sensors apply
cropping to lenses.

An appropriate sensor size is between 2/3" to 1" for balanced high image quality
and versatility in an industrial setting like a recycling facility. This
recommendation considers the following aspects:

-  **Image quality:** Large sensors offer better image quality, higher dynamic
   range, and improved low-light performance.
-  **Field of view (FoV):** Large sensors support a variety of lens options to
   cover the entire width of the conveyor belt.
-  **Depth of field (DoF):** Large sensors achieve a greater DoF, which helps in
   keeping all parts of the objects on the conveyor belt in focus. Use DoF
   calculators like the [DoF simulator](https://dofsimulator.net/en/) to see the
   effect of the [aperture](#aperture-size-f-number) and the sensor size on the
   DoF.

To convert sensor sizes to imaging area dimensions, review external references such as the [Photo Review table of common sensor sizes](https://www.photoreview.com.au/tips/buying/unravelling-sensor-sizes/).

## Focal length

Choose a focal length for the lens depending on the height of the camera from
the conveyor belt and the required FoV. Assuming that the camera height is
fixed, you can calculate how much of the conveyor belt the camera can see based
on the focal length. Up to a certain point, a higher focal length makes it
easier for the model to recognize what the material on an image is. Use the
[sensor size](#sensor-size), camera height, and conveyor belt width to determine
the focal length using the following formula:

_f = (s)(d) / 𝑤_

Where:

-  _f_ is the focal length
-  _s_ is the sensor width (for example, 6 mm assuming a 1/2.3" sensor size)
-  _d_ is the distance from the camera to the conveyor belt (camera height)
-  _𝑤_ is the width of the area to be covered (conveyor belt width)

To simplify the calculation, install the camera at a height that equals the conveyor belt width. For example, suppose you have a [sensor size](#sensor-size) of 1/2.3″ such as the one the [ArduCam IMX477 sensor](https://www.arducam.com/product/b0242-arducam-imx477-hq-camera/) has. This sensor size equals a sensor width of 6 mm. If the conveyor belt width is one meter and you install the camera at one meter above the conveyor belt, then you can use the formula as follows:

_f = (6)(1) / 1 = 6 mm_

Therefore, if the lens's focal length is approximately equal to the sensor
width, you can accommodate different conveyor belt widths by varying the camera
height to be the same as the belt width.

To estimate the focal length of a camera lens, you can use online calculators such as the [ArduCam focal length calculator](https://www.arducam.com/focal-length-calculator/).

## Aperture size (f-number)

The aperture controls the amount of light that gets into the camera sensor and
has an inverse relationship with DoF. A wide aperture gives a shallow depth of
field. On the contrary, a narrow aperture gives you a deeper DoF. Select an
f-number that captures all items in focus as they pass through the conveyor
belt. Choose values between f/2.8 and f/11, depending on the lightning, camera
height and belt width.

## Shutter speed

The shutter speed impacts motion blur and overall brightness. The appropriate
shutter speed depends on the conveyor belt's speed and the objects' motion.
Faster conveyor belts require faster shutter speeds to avoid motion blur. Longer
times on shutter speed let more light in, but motion blur increases. Calculate
how much conveyor belt movement is acceptable while the shutter is open to
prevent motion blurriness.

Use the following formula to estimate shutter speed:

_1 / (2)(frame rate)_

A shutter speed between 1/250 and 1/500 seconds is generally suitable. You can
start with 1/350 seconds to provide a good balance between reducing motion blur
and maintaining image quality. Adjust based on real-world testing and ensure
adequate lighting conditions support these faster shutter speeds.