import tensorflow as tf
import vgg19_trainable as vgg19
import utils

def train(num_class):	
	if True:
		#train		
		with tf.device('/cpu:0'):
			sess = tf.Session()					
			train_mode = tf.constant(True, dtype=tf.bool)
			#读取train的图片和label
			# files_list = "./train_fvgg_emo.txt"
			filename_queue = tf.train.string_input_producer(["./train_fvgg_emo.txt"])
			reader = tf.TextLineReader()
			filename, value= reader.read(filename_queue)
			image_name, label = tf.decode_csv(value, record_defaults=[["string"],["int32"]],field_delim=" ")

			image_content = tf.read_file(image_name)
			image_data = tf.image.decode_jpeg(image_content, channels=3)
			image = tf.image.resize_images(image_data, [224, 224])
			label = tf.string_to_number(label,tf.int32)
			
			labels = tf.one_hot(label,num_class,1,0,dtype=tf.int32)
			labels = tf.cast(labels  , tf.float32)
			image_batch,label_batch = tf.train.batch([image,labels],batch_size = 20)

			vgg = vgg19.Vgg19(num_class,'./vgg19.npy')
			vgg.build(image_batch,train_mode)
			
			cost = tf.reduce_sum((vgg.prob - label_batch) ** 2)
			train = tf.train.AdamOptimizer(1e-4).minimize(cost)
			# coord = tf.train.Coordinator()
			#train
			sess.run(tf.global_variables_initializer())	
			tf.train.start_queue_runners(sess=sess)
			for i in range(5):
				sess.run(train)
				print("train process: %f loss :" % (i/50))
				print(cost)
			#保存模型
			vgg.save_npy(sess,"./rzc_vgg19.npy")
			# if testable:
				#test????????????????????
				# print("{} Start testing".format(datetime.now()))
				# tp = tn = fn = fp = 0
				# for _ in range(200):
				# 	#???????????????????????
				# 	img_batch = sess.run(testing.image_batch)
				# 	label_batch = sess.run(testing.label_batch)
				# 	softmax_prediction = sess.run(score, feed_dict={x: img_batch, y: label_batch})
				# 	prediction_label = sess.run(tf.argmax(softmax_prediction, 1))
				# 	actual_label = sess.run(tf.argmax(label_batch, 1))
				# 	for i in range(len(prediction_label)):
				# 		if prediction_label[i] == actual_label[i] == 1:
				# 			tp += 1
				# 		elif prediction_label[i] == actual_label[i] == 0:
				# 			tn += 1
				# 		elif prediction_label[i] == 1 and actual_label[i] == 0:
				# 			fp += 1
				# 		elif prediction_label[i] == 0 and actual_label[i] == 1:
				# 			fn += 1
				# precision = tp / (tp + fp)
				# recall = tp / (tp + fn)
				# f1 = (2 * tp) / (2 * tp + fp + fn)  # f1为精确率precision和召回率recall的调和平均
				# print("{} Testing Precision = {:.4f}".format(datetime.now(), precision))
# def test_image(path_image, num_class):
# 	#使用训练好的模型测试一张图片
# 	class_name = ['1', '2','3','4','5','6']

# 	img_string = tf.read_file(path_image)
# 	img_decoded = tf.image.decode_png(img_string, channels=3)
# 	img_resized = tf.image.resize_images(img_decoded, [224, 224])
# 	img_resized = tf.reshape(img_resized, shape=[1, 224, 224, 3])
# 	model = Vgg19(bgr_image=img_resized, num_class=num_class, vgg19_npy_path='./vgg19.npy')
# 	score = model.fc8
# 	prediction = tf.argmax(score, 1)
# 	saver = tf.train.Saver()
# 	with tf.Session() as sess:
# 		sess.run(tf.global_variables_initializer())
# 		saver.restore(sess, "./model_epoch50.ckpt")
# 		plt.imshow(img_decoded.eval())
# 		plt.title("Class:" + class_name[sess.run(prediction)[0]])
# 		plt.show()
# test_image('./validate/11.jpg', 2)
train(6)