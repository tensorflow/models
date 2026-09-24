
import tensorflow as tf
from absl.testing import parameterized
from official.projects.fcos.model.model import FCOS

class FCOSModelTest(parameterized.TestCase, tf.test.TestCase):

    def test_fcos_model_instantiation_and_forward_pass(self):
        """Tests that the FCOS model can be instantiated and run."""
        
        # Create the model
        model = FCOS()
        
        # specific input shape [Batch, H, W, 3]
        input_image_size = (1, 800, 1024, 3)
        inputs = tf.random.normal(input_image_size)
        
        # Run forward pass
        outputs = model(inputs, training=False)
        
        # Check output keys
        self.assertIn('classifier', outputs)
        self.assertIn('box', outputs)
        self.assertIn('centerness', outputs)
        
        # Check output shapes (approximate checks based on feature map sizes)
        # We expect a flattened output of shape [Batch, N_points, Channels]
        self.assertEqual(outputs['classifier'].shape[0], 1)
        self.assertEqual(outputs['box'].shape[0], 1)
        self.assertEqual(outputs['centerness'].shape[0], 1)
        
        # Detailed shape check for channels
        self.assertEqual(outputs['classifier'].shape[-1], 80) # 80 classes
        self.assertEqual(outputs['box'].shape[-1], 4)       # 4 regression coords
        self.assertEqual(outputs['centerness'].shape[-1], 1)# 1 centerness score

if __name__ == '__main__':
    tf.test.main()
