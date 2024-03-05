
      '''
      # for instance segmentation
      memory4 = tf.reshape(memory4, [batch_size, 4, 4, self.dim])
      bbox_mask4 = self.bbox_attention_hs4(hs4, memory4) 

      memory3 = tf.reshape(memory3, [batch_size, 8, 8, self.dim])
      bbox_mask3 = self.bbox_attention_hs3(hs3, memory3) 

      memory2 = tf.reshape(memory2, [batch_size, 16, 16, self.dim])
      bbox_mask2 = self.bbox_attention_hs2(hs2, memory2) 
      
      memory1 = tf.reshape(memory1, [batch_size, 32, 32, self.dim])
      bbox_mask1 = self.bbox_attention_hs1(hs1, memory1) 

      
      # for instancd_segmentation
      #c5 : [b, 4, 4, 2048] 
      #bbox4 : [b, 10, 8, 4, 4]
      b_shape = bbox_mask4.shape
      bbox_mask4 = tf.reshape(bbox_mask4, [batch_size, b_shape[1]*b_shape[2], b_shape[3], b_shape[4]])
      bbox_mask4 = tf.transpose(bbox_mask4, perm = [0, 2, 3, 1]) 
      x = tf.concat([c5, bbox_mask4] , -1)
      #x = tf.concat([_expand(c5, bbox_mask4.shape[3]), bbox_mask4.flatten(0, 1)], -1)' 
      x = self.lay1(x)
      x = self.gn1(x)
      x = self.relu(x)
      x = self.lay2(x)
      x = self.gn2(x)
      x = self.relu(x)
      
      b_shape = bbox_mask3.shape
      bbox_mask3 = tf.reshape(bbox_mask3, [batch_size, b_shape[1]*b_shape[2], b_shape[3], b_shape[4]])
      bbox_mask3 = tf.transpose(bbox_mask3, perm = [0, 2, 3, 1]) 
      fpn1_input = tf.concat([c4, bbox_mask3] , -1)

      cur_fpn = self.adapter1(fpn1_input) # [64, 8, 8, 72]
      x = cur_fpn + tf.image.resize(x, cur_fpn.shape[1:3], method=tf.image.ResizeMethod.BILINEAR) 
      x = self.lay3(x)
      x = self.gn3(x)
      x = self.relu(x)

      b_shape = bbox_mask2.shape
      bbox_mask2 = tf.reshape(bbox_mask2, [batch_size, b_shape[1]*b_shape[2], b_shape[3], b_shape[4]])
      bbox_mask2 = tf.transpose(bbox_mask2, perm = [0, 2, 3, 1]) 
      fpn2_input = tf.concat([c3, bbox_mask2] , -1)

      cur_fpn = self.adapter2(fpn2_input)
      x = cur_fpn + tf.image.resize(x, cur_fpn.shape[1:3], method=tf.image.ResizeMethod.BILINEAR) 
      x = self.lay4(x)
      x = self.gn4(x)
      x = self.relu(x)

      b_shape = bbox_mask1.shape
      bbox_mask1 = tf.reshape(bbox_mask1, [batch_size, b_shape[1]*b_shape[2], b_shape[3], b_shape[4]])
      bbox_mask1 = tf.transpose(bbox_mask1, perm = [0, 2, 3, 1]) 
      fpn3_input = tf.concat([c2, bbox_mask1] , -1)

      cur_fpn = self.adapter3(fpn3_input)
      x = cur_fpn + tf.image.resize(x, cur_fpn.shape[1:3], method=tf.image.ResizeMethod.BILINEAR) 
      x = self.lay5(x)
      x = self.gn5(x)
      x = self.relu(x)

      x = self.out_lay(x)
      x = tf.image.resize(x, size=(self.output_size, self.output_size) , method=tf.image.ResizeMethod.BILINEAR)'''