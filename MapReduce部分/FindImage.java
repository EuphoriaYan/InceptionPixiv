/**
 * @program: FindImage
 * @description:
 * @author: Euphoria Yan
 * @create: 2019-01-22 17:07
 **/


import java.io.*;
import java.util.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FindImage {

    /* Map的输出的Reduce的输入必须是一个实现了Writable接口的类
     * 自定义一个类作为Map输出的value类
     * 包含角度，文件夹，文件名，重载write和readFields函数
     */
    public static class MyWritable implements Writable {
        private float angle;
        private int folder;
        private int document;

        public float getAngle() {
            return angle;
        }

        public void setAngle(float angle) {
            this.angle = angle;
        }

        public int getFolder() {
            return folder;
        }

        public void setFolder(int folder) {
            this.folder = folder;
        }

        public int getDocument() {
            return document;
        }

        public void setDocument(int document) {
            this.document = document;
        }

        public MyWritable() {
        }
        public MyWritable(float angle, int folder, int document){
            this.angle = angle;
            this.folder = folder;
            this.document = document;
        }
        public MyWritable(MyWritable b){
            this.angle = b.getAngle();
            this.folder = b.getFolder();
            this.document = b.getDocument();
        }
        @Override
        public String toString() {
            return this.angle+","+this.folder+","+this.document;
        }
        @Override
        public void write(DataOutput out) throws IOException {
            out.writeFloat(angle);
            out.writeInt(folder);
            out.writeInt(document);
        }
        @Override
        public void readFields(DataInput in) throws IOException {
            this.angle = in.readFloat();
            this.folder = in.readInt();
            this.document = in.readInt();
        }

    }

    // Map类
    public static class FindMapper extends Mapper<Object, Text, IntWritable, MyWritable> {

        public Logger log = LoggerFactory.getLogger(FindMapper.class);
        private List<float[]> testData;  // 保存测试集，每一个测试点由2048个float值组成，因此是float[]，多个测试点以List形式组成测试集

        // setup类是只在一开始运行一次
        @Override
        protected void setup(Context context)
                throws IOException, InterruptedException {
            Configuration conf= context.getConfiguration();  // 读取整个MapReduce公用的配置文件conf
            String testPath = conf.get("test_path");  // 从conf中读取测试文件在HDFS中的路径
            Path testDataPath= new Path(testPath);  // 转换成Path类型
            FileSystem fs = FileSystem.get(conf);  // 获得HDFS文件系统对象
            this.testData = readTestData(fs, testDataPath);  // 读取测试文档
        }

        /* Map对于每条记录运行一次，这里的“每条记录”实际上是训练点
         * Map输入的 key => 文件中的偏移值，不关心
         * Map输入的 value => 一个训练点的数据
         * Map输出的 key => 测试点id
         * Map输出的 value => {和当前训练点的余弦值，训练点文件夹，训练点文件名}
         * 一共输出“测试点个数”个键值对。
         */
        @Override
        protected void map(Object key, Text value,
                           Mapper<Object, Text, IntWritable, MyWritable>.Context context)
                throws IOException, InterruptedException {
            String[] line = value.toString().split(",");  // 读入一条记录，key是什么不关心，只需要value，也就是训练点
            float[] trainData = new float[line.length-2];  // 建立一个2048长度的一维float数组，存放该训练点的点值。
            int folder = Integer.valueOf(line[0]);  // 分割出的第一个值是folder，文件夹名
            int document = Integer.valueOf(line[1]);  // 分割出的第二个值是document，图片编号
            for(int i = 1; i < trainData.length; i++) {
                trainData[i] = Float.valueOf(line[i]);  // 依次将分割出的String转换为float，并存到trainData中
            }
            for(int i = 0; i < this.testData.size(); i++) {
                float[] testI = this.testData.get(i);  // 读取第i个测试点
                float distance = calCos(testI, trainData);  // 计算当前训练点和第i个测试点的欧拉距离
                // log.info("距离："+distance);
                // Map的输出格式是 <测试点编号i，{和当前训练点的余弦值，训练点文件夹，训练点文件名}>
                context.write(new IntWritable(i), new MyWritable(distance, folder, document));  //写入Map的输出中
            }
        }

        // 用来读测试文件，hdfs中一样读就行了，并不特殊
        private List<float[]> readTestData(FileSystem fs,Path Path) throws IOException {
            FSDataInputStream data = fs.open(Path);
            BufferedReader bf = new BufferedReader(new InputStreamReader(data));
            String line = "";
            List<float[]> list = new ArrayList<>();
            while ((line = bf.readLine()) != null) {
                String[] items = line.split(",");
                float[] item = new float[items.length];
                for(int i=0;i<items.length;i++){
                    item[i] = Float.valueOf(items[i]);
                }
                list.add(item);
            }
            return list;
        }

        // 计算夹角cos
        private static float calCos(float[] testData, float[] inData) {
            float ab =0.0f;
            float a = 0.0f;
            float b = 0.0f;
            for(int i = 0;i < testData.length; i++){
                ab += testData[i] * inData[i];
                a += testData[i] * testData[i];
                b += inData[i] * inData[i];
            }
            a = (float)Math.sqrt(a);
            b = (float)Math.sqrt(b);
            if (Math.abs(a)<1e9 | Math.abs(b)<1e9)
                return 1;
            ab = ab / (a * b);
            return ab;
        }
    }

    // Reduce类
    public static class FindReducer extends Reducer<IntWritable, MyWritable, IntWritable, Text> {

        /*
         * Map输出的 key => 测试点id
         * Map输出的 value => {和某个训练点的余弦值，训练点文件夹，训练点文件名}
         * Reduce输出的 key => 测试点id
         * Reduce输出的 value => 最相似的图片位置
         */
        @Override
        protected void reduce(IntWritable key, Iterable<MyWritable> values,
                              Reducer<IntWritable, MyWritable, IntWritable, Text>.Context context) throws IOException, InterruptedException {
            MyWritable res = new MyWritable(-2, 1, 1);  // 最大值，cos值为-1~1，所以设个-2

            /* 更新最小值
             */
            for (MyWritable m : values) {
                if (m.getAngle()>res.getAngle()){
                    res.setAngle(m.getAngle());
                    res.setFolder(m.getFolder());
                    res.setDocument(m.getDocument());
                }
            }
            String Result = res.getDocument() + " " + res.getFolder();
            // Reduce的输出 <测试点编号， 结果>
            context.write(key, new Text(Result));
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();//MR的配置类
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        // 参数是训练文件路径，测试文件路径，输出文件路径
        if (otherArgs.length != 3) {
            System.err.println("Usage: FindImage <train> <test> <output>");
            System.exit(2);
        }
        conf.set("test_path", args[1]);
        Job job = Job.getInstance(conf, "FindImage");//新建一个job
        job.setJarByClass(KnnClassify.class);//job对应的class
        job.setMapperClass(FindMapper.class);//job的mapper
        job.setMapOutputKeyClass(IntWritable.class);//Map输出的key类型
        job.setMapOutputValueClass(MyWritable.class);//Map输出的value类型
        job.setReducerClass(FindReducer.class);//job的reducer
        job.setOutputKeyClass(IntWritable.class);//输出key
        job.setOutputValueClass(Text.class);//输出value
        FileInputFormat.addInputPath(job, new Path(otherArgs[0]));//train文件作为输入
        FileOutputFormat.setOutputPath(job, new Path(otherArgs[2]));//输出文件
        System.exit(job.waitForCompletion(true) ? 0 : 1); //job的退出条件
    }
}