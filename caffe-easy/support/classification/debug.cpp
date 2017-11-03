#include <Windows.h>
#include "classification.h"
#include <highgui.h>
#include <process.h>
#define RootDir "../../../../../demo-data/veriCode/"
#include <map>
//#define RootDir

using namespace cv;
using namespace std;


#if 0
vector<char> readAtFile(const char* filename){
	FILE* f = fopen(filename, "rb");
	if (f == 0) return vector<char>();

	fseek(f, 0, SEEK_END);
	int len = ftell(f);
	fseek(f, 0, SEEK_SET);

	vector<char> buf(len);
	fread(&buf[0], 1, len, f);
	fclose(f);
	return buf;
}

map<int, int> mp;
CRITICAL_SECTION gcs;
void recThread(void* param){
	TaskPool* pool = (TaskPool*)param;
	vector<char> imd = readAtFile(RootDir "samples/0A79_xxx.png");

	double time = getTickCount();
	for(int i = 0; i < 1000000; ++i){
		int labels[4];
		float confs[4];
		time = getTickCount();

		if (i % 2 == 0){
			SoftmaxResult* val = predictSoftmaxByTaskPool(pool, &imd[0], imd.size(), 1);

			time = (getTickCount() - time) / getTickFrequency() * 1000.0;
			getMultiLabel(val, labels);
			getMultiConf(val, confs);
#if 1
			//if (i % 100 == 0 && val && blob){

			//printf("0x%p labels = %d, %d, %d, %d\n", val, labels[0], labels[1], labels[2], labels[3]);
			printf("0x%p confs = %f, %f, %f, %f\n", val, confs[0], confs[1], confs[2], confs[3]);
			//}
			//printf("%.2f, 耗时：%.2f ms\n", getTickCount() / getTickFrequency(), time);
#endif
			if (val == 0){
				printf("error\n");
				exit(0);
			}
			//printf("%d\n", val);
			releaseSoftmaxResult(val);
		}
		else{
			BlobData* blob = forwardByTaskPool(pool, &imd[0], imd.size(), "cccp7");
			if (blob == 0){
				printf("error\n");
				exit(0);
			}
			printf("0x%p blob = %d, %d, %d, %d\n", blob, blob->count, blob->channels, blob->height, blob->width);
			releaseBlobData(blob);
		}

#if 0
		EnterCriticalSection(&gcs);
		if (mp.find((int)val) != mp.end()){
			printf("error.\n");
			assert(false);
			exit(0);
		}
		mp[(int)val] = 1;
		LeaveCriticalSection(&gcs);
#endif
	}
}

#if 1
void main(int argc, char** argv){

#if 0
	Classifier c(RootDir "deploy.prototxt", RootDir "nin_iter_16000.caffemodel", 1.0, "", 0, 0, -1, 10);

	int num = argc > 1 ? atoi(argv[1]) : 1;
	Mat im = imread(RootDir "samples/00W0_27c86a8b9ce8d0b1fe1d3d47b4040a28.png");
	Mat im1 = imread(RootDir "samples/0FAW_a103991142caf37bfc7912c7cd2162b9.png");

	//单图测试
#if 0
	WPtr<SoftmaxResult> softmax = c.predictSoftmax(im, 1);
	int labels[4];
	float confs[4];
	getMultiLabel(softmax, labels);
	getMultiConf(softmax, confs);
	printf("labels = %d, %d, %d, %d\n", labels[0], labels[1], labels[2], labels[3]);
	printf("confs = %f, %f, %f, %f\n", confs[0], confs[1], confs[2], confs[3]);
#endif

	///测试多图传入问题
#if 0
	vector<Mat> imgs;
	for (int i = 0; i <num; ++i){
		imgs.push_back(i % 2 == 0 ? im : im1);
	}

	double tck = cv::getTickCount();
	WPtr<MultiSoftmaxResult> softmax = c.predictSoftmax(imgs, 1);
	tck = (cv::getTickCount() - tck) / cv::getTickFrequency() * 1000.0;
	printf("耗时：%.2f ms\n", tck);
	for (int i = 0; i < softmax->count; ++i){
		int labels[4];
		float confs[4];
		SoftmaxResult* val = softmax->list[i];
		getMultiLabel(val, labels);
		getMultiConf(val, confs);
		printf("labels = %d, %d, %d, %d\n", labels[0], labels[1], labels[2], labels[3]);
		printf("confs = %f, %f, %f, %f\n", confs[0], confs[1], confs[2], confs[3]);
	}
#endif
#endif

	InitializeCriticalSection(&gcs);

	//测试任务池
	TaskPool* pool = createTaskPool(RootDir "deploy.prototxt", RootDir "nin_iter_16000.caffemodel", 1.0, "", 0, 0, 0, 32);
	for (int i = 0; i < 160; ++i){
		_beginthread(recThread, 0, pool);
	}
	Sleep(1000 * 100000);
	printf("停止...\n");
	releaseTaskPool(pool);
	printf("已经停止...\n");
	Sleep(3000);
}
#endif

#if 0
typedef int(__stdcall *procCCTrainEventCallback)(int event, int param1, float param2, void* param3);
extern void setTrainEventCallback(procCCTrainEventCallback callback);
extern "C" Caffe_API void __stdcall setTraindEventCallback(procCCTrainEventCallback callback);
extern "C" Caffe_API int __stdcall train_network(char* args);

int __stdcall testx(int event, int param1, float param2, void* param3){
	printf("event %d:\n", event);
	if (event == 7){
		char* p = (char*)param3;
		p += 16;
		char** ptr = *(char***)p;
		printf("%s\n", ptr[0]);
	}
	return 0;
}
void main(){
	setTraindEventCallback(testx);

	string info = "train --solver=solver.prototxt";
	train_network((char*)info.c_str());
}
#endif
#endif


Scalar getColor(int label){
	static vector<Scalar> colors;
	if (colors.size() == 0){
#if 0
		for (float r = 127.5; r <= 256 + 127.5; r += 127.5){
			for (float g = 256; g >= 0; g -= 127.5){
				for (float b = 0; b <= 256; b += 127.5)
					colors.push_back(Scalar(b, g, r > 256 ? r - 256 : r));
			}
		}
#endif
		colors.push_back(Scalar(255, 0, 0));
		colors.push_back(Scalar(0, 255, 0));
		colors.push_back(Scalar(0, 0, 255));
		colors.push_back(Scalar(0, 255, 255));
		colors.push_back(Scalar(255, 0, 255));
		colors.push_back(Scalar(128, 0, 255));
		colors.push_back(Scalar(128, 255, 255));
		colors.push_back(Scalar(255, 128, 255));
		colors.push_back(Scalar(128, 255, 128));
	}
	return colors[label % colors.size()];
}

#define VIDEO_FILE0 "H:/work/deepblue/2.7.28行为模拟视频/longtime/000002-cam0.mp4"
#define VIDEO_FILE1 "H:/work/deepblue/2.7.28行为模拟视频/4/000003-cam0.mp4"
#define VIDEO_FILE2 "H:/work/deepblue/2.7.28行为模拟视频/放回去实验/000000-cam0.mp4"
#define VIDEO_FILE3 "H:/work/deepblue/2.7.28行为模拟视频/3/000002-cam0.mp4"

int g_sd_line = 300;		//S和D区分割线
int g_dg_line = 495;		//D和G区分割线


struct DetectObjectInfo
{
	int image_id;
	int label;
	float score;

	int xmin;
	int ymin;
	int xmax;
	int ymax;

	const Rect rect() const
	{
		return Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
	}
};

vector<DetectObjectInfo> toDetInfo(BlobData* fr, int imWidth = 1, int imHeight = 1, float threshold = 0.5){
	vector<DetectObjectInfo> out;
	if (!fr) return out;

	float* data = fr->list;
	for (int i = 0; i < fr->count; i += 7, data += 7)
	{
		//if invalid det
		if (data[0] == -1 || data[2] < threshold)
			continue;

		DetectObjectInfo obj;
		obj.image_id = data[0];
		obj.label = data[1];
		obj.score = data[2];
		obj.xmin = data[3] * imWidth;
		obj.ymin = data[4] * imHeight;
		obj.xmax = data[5] * imWidth;
		obj.ymax = data[6] * imHeight;
		out.push_back(obj);
	}
	return out;
}

#if 1
void main(){
	disableErrorOutput();

	const char* video_files[] = { VIDEO_FILE0, VIDEO_FILE1, VIDEO_FILE2, VIDEO_FILE3 };
	float means[] = { 104.0f, 117.0f, 123.0f };
	Classifier cc("H:/work/deepblue/1.lg/4.camera-test/bin/new-model-resnet/deploy.prototxt", "H:/work/deepblue/1.lg/4.camera-test/bin/new-model-resnet/model.caffemodel", 1, 0, 3, means, 0, 5);
	//Classifier cc("C:/Users/Administrator/Desktop/deploy.prototxt.txt", "C:/Users/Administrator/Desktop/_iter_122659.caffemodel", 1, 0, 0, 0, 0, 5);
	string path = "H:/work/deepblue/1.lg/4.camera-test/bin/";
	string imps[] = { path + "camera 0.jpg", path + "camera 1.jpg", path + "camera 2.jpg" };
	//Mat ims[] = { imread(imps[0]), imread(imps[1]), imread(imps[2]) };

	VideoCapture caps[4];
	int flags[4] = { 0 };

	for (int i = 0; i < 4; ++i){
		caps[i].open(video_files[i]);
		caps[i].set(CV_CAP_PROP_FRAME_WIDTH, 800);
		caps[i].set(CV_CAP_PROP_FRAME_HEIGHT, 600);
	}

	int area_w = g_dg_line - g_sd_line;

	while (true){

		vector<Mat> ms;
		for (int i = 0; i < 4; ++i){
			if (flags[i] == 1) continue;
			Mat frame;
			caps[i] >> frame;
			if (!frame.empty())
				ms.push_back(frame);
			else
				flags[i] = 1;
		}

		if (ms.size() == 0) break;

		Mat ssdinput(ms[0].rows, area_w*ms.size(), CV_8UC3);
		for (int i = 0; i < ms.size(); ++i){
			ms[i](Rect(g_sd_line, 0, area_w, ms[i].rows)).copyTo(ssdinput(Rect(area_w*i, 0, area_w, ms[i].rows)));
		}

		//cc.reshape(ms.size(), -1, -1);
		cc.forward(ssdinput);

		WPtr<BlobData> data0 = cc.getBlobData(0, "detection_out");
		auto objs = toDetInfo(data0, ssdinput.cols, ssdinput.rows);
		for (auto obj : objs){
			rectangle(ssdinput, obj.rect(), getColor(obj.label), 2);
		}
		imshow("ssdinput", ssdinput);
		waitKey(1);
	}
	//WPtr<BlobData> data1 = cc.getBlobData(1, "detection_out");
	//WPtr<BlobData> data0 = cc.getBlobData(0, "premuted_fc");
	//WPtr<BlobData> data1 = cc.getBlobData(1, "premuted_fc");
}
#endif


#if 0
void main(){
	disableErrorOutput();

	const char* video_files[] = { VIDEO_FILE0, VIDEO_FILE1, VIDEO_FILE2, VIDEO_FILE3 };
	float means[] = { 104.0f, 117.0f, 123.0f };
	Classifier cc("H:/work/deepblue/1.lg/4.camera-test/bin/new-model-resnet/deploy.prototxt", "H:/work/deepblue/1.lg/4.camera-test/bin/new-model-resnet/model.caffemodel", 1, 0, 3, means, 0, 5);
	//Classifier cc("C:/Users/Administrator/Desktop/deploy.prototxt.txt", "C:/Users/Administrator/Desktop/_iter_122659.caffemodel", 1, 0, 0, 0, 0, 5);
	string path = "H:/work/deepblue/1.lg/4.camera-test/bin/";
	string imps[] = { path + "camera 0.jpg", path + "camera 1.jpg", path + "camera 2.jpg" };
	//Mat ims[] = { imread(imps[0]), imread(imps[1]), imread(imps[2]) };

	VideoCapture caps[4];
	for (int i = 0; i < 4; ++i)
		caps[i].open(video_files[i]);

	while (true){

		vector<Mat> ms;
		for (int i = 0; i < 4; ++i){
			Mat frame;
			caps[i] >> frame;
			if (!frame.empty())
				ms.push_back(frame);
		}

		if (ms.size() == 0) break;

		cc.reshape(ms.size(), -1, -1);
		cc.forward(&ms[0], ms.size());

		WPtr<BlobData> data0 = cc.getBlobData(0, "deepblue_out");
		float* p = data0->list;
		int num = *p++;
		for (int i = 0; i < num; ++i){
			int numobj = *p++;
			for (int j = 0; j < numobj; ++j){
				int numbox = *p++;
				for (int b = 0; b < numbox; ++b){
					int image_id = *p++;
					int label = *p++;
					float score = *p++;

					float xmin = *p++ * ms[i].cols;
					float ymin = *p++ * ms[i].rows;
					float xmax = *p++ * ms[i].cols;
					float ymax = *p++ * ms[i].rows;
					if (image_id == -1 || score <= 0.8)
						continue;

					rectangle(ms[i], Rect(xmin, ymin, xmax - xmin, ymax - ymin), getColor(label), 2);
				}
			}

			char buf[100];
			sprintf(buf, "camera %d", i);
			imshow(buf, ms[i]);
		}

		waitKey(1);
	}
	WPtr<BlobData> data1 = cc.getBlobData(1, "detection_out");
	//WPtr<BlobData> data0 = cc.getBlobData(0, "premuted_fc");
	//WPtr<BlobData> data1 = cc.getBlobData(1, "premuted_fc");
}
#endif