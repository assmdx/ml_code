import tensorflow as tf

VGG_MEAN = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)



class ImageDataGenerator(object):
    def __init__(self, images, labels, batch_size, num_classes):
        self.filenames = images
        self.labels = labels
        self.batch_size = batch_size
        self.num_class = num_classes
        self.image_batch, self.label_batch = self.image_decode()


    def image_decode(self):
        # 建立文件队列，把图片和对应的实际标签放入队列中
        #注：在没有运行tf.train.start_queue_runners(sess=sess)之前，数据实际上是没有放入队列中的
        file_queue = tf.train.slice_input_producer([self.filenames, self.labels])

        # 把图片数据转化为三维BGR矩阵
        image_content = tf.read_file(file_queue[0])
        image_data = tf.image.decode_jpeg(image_content, channels=3)
        image = tf.image.resize_images(image_data, [224, 224])
        img_centered = tf.subtract(image, VGG_MEAN)
        img_bgr = img_centered[:, :, ::-1]

        labels = tf.one_hot(file_queue[1],self.num_class, dtype=tf.uint8)

        # 分batch从文件队列中读取数据
        image_batch, label_batch = tf.train.shuffle_batch([img_bgr, labels],
                                                          batch_size=self.batch_size,
                                                          capacity=2000,
                                                          min_after_dequeue=1000)        
        return image_batch, label_batch