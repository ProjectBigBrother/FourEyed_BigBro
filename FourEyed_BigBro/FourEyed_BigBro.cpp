#include "stdafx.h"
#include "OpenCV_Headers.h"

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels);
int Distance(Point p1, Point p2);


int main(int argc, const char *argv[]) {

	int mode = 1;
	cout << "WELCOME!" << endl;
	//SELECT MODE
	//cout << "Select Mode: Normal=0; Learn!=0";
	//cin >> mode;



	VideoCapture cap(0);
	if (!cap.isOpened())
		return -1;
	cout << "Camera Stream Added" << endl;
	string fn_haar = "C:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml";
	string fn_haar_eyes = "C:/opencv/sources/data/haarcascades/haarcascade_mcs_lefteye.xml";//haarcascade_eye_tree_eyeglasses.xml";
	string fn_csv = "C:/Users/Ian/Desktop/FourEyed_BigBro/FourEyed_BigBro/Database/Database.txt";
	
	cout << "Load images:: " << endl;
	vector<Mat> images;
	vector<Mat> formatedImages;
	vector<int> labels;
	try {
		read_csv(fn_csv, images, labels);
	}
	catch (cv::Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		// nothing more we can do
		while (true){
			if (!waitKey(0))
				break;
		}
		exit(1);
	}
	cout << "::Images Loaded:: "<< images.size()<<"::"<<labels.size()<< endl;
	
	for (int x = 0; x < images.size(); x++){
		resize(images[x], images[x], Size(120, 90), 0, 0, INTER_CUBIC);
		imshow("image",images[x]);

		/*Rect lock(0, 30, 120, 30);
		formatedImages.push_back(images[x](lock));
		imshow("image_resized", formatedImages[x]);
		cout << formatedImages[x].size() << "  Label:: " << labels[x] << endl;*/

		waitKey(1);
	}

	cout << "Creating Model::" << endl;
	Ptr<FaceRecognizer> FourEyes = createLBPHFaceRecognizer();//createFisherFaceRecognizer(); // createEigenFaceRecognizer();
	FourEyes->train(images, labels);//formatedImages, labels);
	
	cout << "Creating FaceDetector::" << endl;
	CascadeClassifier myFaceDetector;
	myFaceDetector.load(fn_haar);

	/*
	cout << "Create Eye Detector" << endl;
	CascadeClassifier myEyeDetector;
	myEyeDetector.load(fn_haar_eyes);
	*/

	cout << "::Training Complete" << endl;
	
	cout << "Creating Variables" << endl;
	Mat frame, original, grey;
	vector< Rect > faces;
	vector< Rect > eyes;
	Mat face_resized;
	Mat face_im; 

	cout << "Entering Main Loop" << endl;
	while (true) {
		cap >> frame;
		
		Mat original = frame.clone();
		
		cvtColor(original, grey, CV_BGR2GRAY);
		
		myFaceDetector.detectMultiScale(grey, faces);
		

		cout << "FACES: " << faces.size() << endl;
		if (faces.size() > 0){

			if (mode == 1){
				
				cout << "Select Mode: 2=train; Run=0; Database=3;";
				cin >> mode;
			}
			else if (mode == 2){
				cout << "Creating Model::" << endl;
				if (images.size() > 0){
					int model = 0;
					cout << "Select Model: 0=Eigen; 1=Fisher; 2=LBP";
					cin >> model;
					switch (model){
					case 0:
						FourEyes = createEigenFaceRecognizer();
					case 1:
						FourEyes = createFisherFaceRecognizer();
					case 2:
						FourEyes = createLBPHFaceRecognizer();
					default:
						FourEyes = createEigenFaceRecognizer();
					}

					//Ptr<FaceRecognizer> FourEyes = createEigenFaceRecognizer();
					FourEyes->train(images, labels);
					mode = 1;
				}
				else{
					cout << "Model is not trained" << endl;
				}
			}
			else if (mode==3){
				int op; int label;
				cout << "Select Mode: 0=replace  1=add";
				cin >> op;
				if (op == 0){
					images.clear();
					labels.clear();
				}
				cout << "Enter Label: 4=glasses, 2=no Glasses";
				cin >> label;
				for (int y = 0; y < 10; y++){

					cap >> frame;
					original = frame.clone();
					cvtColor(original, grey, CV_BGR2GRAY);
					myFaceDetector.detectMultiScale(grey, faces);

					for (int i = 0; i < faces.size(); i++){
						Rect face_coord = faces[i];
						face_im = grey(face_coord);
						resize(face_im, face_im, Size(120, 90), 1.0, 1.0, INTER_CUBIC);
						imshow("face_resized", face_im);

						/*Rect lock(0, 30, 120, 30);
						face_resized = face_im(lock);

						//cout << face_resized.size() << endl;
						imshow("eyes", face_resized);
						*/
						images.push_back(face_im);//face_resized);
						labels.push_back(label);
						waitKey(10);
					}
				}

				mode = 1;
				//cin >> op;
			}

			else if (mode==0 && images.size()>0){
				for (int i = 0; i < faces.size(); i++){
					Rect face_coord = faces[i];
					face_im = grey(face_coord);
					resize(face_im, face_im, Size(120, 90), 1.0, 1.0, INTER_CUBIC);
					//imshow("face_resized", face_im);

					Rect lock(0, 30, 120, 30);
					face_resized = face_im(lock);

					//cout << face_resized.size() << endl;
					//imshow("eyes",face_resized);

					int glassesPred = FourEyes->predict(face_im);// face_resized);

					String pred = "Unknown";
					if (glassesPred == 4){
						pred = "Glasses";
					}
					else if (glassesPred == 2){
						pred = "No Glasses";
					}
					else(pred = "Unknown");

					cout << "Predicted: " << pred << endl;


					// Calculate the position for annotated text (make sure we don't
					// put illegal values in there):

					/*vector<Rect> eyes_centers;

					myFaceDetector.detectMultiScale(face_im, eyes);
					cout << "EYES: " << eyes.size() << endl;
					if (eyes.size() >0){
					for (int y = 0; y < eyes.size(); y++){
					Rect eye_coord = eyes[y];
					//Point center(faces[i].x + eye_coord.x + eye_coord.width*0.5, faces[i].y + eye_coord.y + eye_coord.height*0.5);
					eyes_centers.push_back(eye_coord);
					}
					//int dist = Distance(eyes_centers[0], eyes_centers[1]);

					try{
					Rect of_interest(eyes_centers[0].x-5, eyes_centers[0].y, eyes_centers[0].width+5, eyes_centers[0].height/2);
					face_resized = face_im(of_interest);
					imshow("eyes", face_resized);
					}
					catch (int x){
					cout << "bad resize" << endl;
					};

					//resize(face_im, face_resized, Size(dist, dist), 0, 0, INTER_CUBIC);

					} */

					rectangle(original, face_coord, CV_RGB(0, 255, 0), 1);
					int pos_x = std::max(face_coord.tl().x - 10, 0);
					int pos_y = std::max(face_coord.tl().y - 10, 0);
					// And now put it into the image:
					putText(original, pred, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 2.0);
					imshow("Original", original);
					
					int c = waitKey(1);
					if (c==27) { 
						mode = 1; 
					}
				

				}
			}
		}

		//waitKey(1);
	}
	return 0;
}

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels) {
	char separator = ';';
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "#1 No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;// classlabel2;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel, separator);
		getline(liness, classlabel, separator);
		getline(liness, classlabel);
		//getline(liness, classlabel2);
		if (!path.empty() && !classlabel.empty()) {
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
			//Gender_labels.push_back(atoi(classlabel2.c_str()));
		}
		//cout << classlabel << endl;
	}
}

int Distance(Point p1, Point p2) {
	int dx = p2.x - p1.x;
	int dy = p2.y - p1.y;
	return (int)sqrt(dx*dx + dy*dy);
}


/*

Mat ScaleRotateTranslate(Mat image, int angle, Point2d center, Poinnew_center, double scale, resample = Image.BICUBIC){
	if (scale is None) and(center is None) {
		return image.rotate(angle = angle, resample = resample)
	}
	nx, ny = x, y = center;
	sx = sy = 1.0;
	if {new_center(nx, ny) = new_center){
		if (scale(sx, sy) = (scale, scale))
			cosine = math.cos(angle);
		sine = math.sin(angle);
		a = cosine / sx;
		b = sine / sx;
		c = x - nx*a - ny*b;
		d = -sine / sy;
		e = cosine / sy;
		f = y - nx*d - ny*e;
		return image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f), resample = resample);
	}

	void CropFace(image, eye_left = (0, 0), eye_right = (0, 0), offset_pct = (0.2, 0.2), dest_sz = (70, 70)) :
		// calculate offsets in original image
		offset_h = math.floor(float(offset_pct[0])*dest_sz[0]);
	offset_v = math.floor(float(offset_pct[1])*dest_sz[1]);
		// get the direction
	eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1]);
		// calc rotation angle in radians
	rotation = -math.atan2(float(eye_direction[1]), float(eye_direction[0]));
		// distance between them
	dist = Distance(eye_left, eye_right);
		// calculate the reference eye - width
	reference = dest_sz[0] - 2.0*offset_h;
		// scale factor
	scale = float(dist) / float(reference);
#//rotate original around the left eye
	image = ScaleRotateTranslate(image, center = eye_left, angle = rotation);
#//crop the rotated image
	crop_xy = (eye_left[0] - scale*offset_h, eye_left[1] - scale*offset_v);
	crop_size = (dest_sz[0] * scale, dest_sz[1] * scale);
	image = image.crop((int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0] + crop_size[0]), int(crop_xy[1] + crop_size[1])));
		// resize it
	image = image.resize(dest_sz, Image.ANTIALIAS);
		return image;
	}
	*/