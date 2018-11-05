#pragma once
#include <vector>
#include <stdlib.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <windows.h>


using namespace std;
using namespace cv;

template <class T>
T pow_m(T ch, double dig, double max);


class Cluster {
public:
	vector<POINT> scores;


	bool active_flag;

	Cluster()
	{
		active_flag = true;
		
	}


	int mid_r;
	int mid_g;
	int mid_b;



	int curX, curY;//координаты текущего центроида
	int lastX, lastY;//координаты предыдущего центоида
	size_t Size() { return scores.size(); }//получаем размер вектора
	inline void Add(POINT pt) { scores.push_back(pt); }//Добавляем пиксель к кластеру
	void SetCenter();
	void Clear();//Чистим вектор
	static Cluster* Bind(int k, Cluster * clusarr, vector<POINT>& vpt, Mat &src, double c_dig, double d_dig, double c_coef, double d_coef, double max);
	static void InitialCenter(int k, Cluster * clusarr, vector<POINT>& vpt, Mat &src);
	Mat Start(int k, Cluster * clusarr, vector<POINT>& vpt, Mat &src, double c_dig, double d_dig, double c_coef, double d_coef, double max);
	inline POINT& at(unsigned i) { return scores.at(i); }//Доступ  к элементам вектора



	void resetMidColor(Mat &src)
	{
		if (scores.size() == 0) return;

		int sum_r = 0;
		int sum_g = 0;
		int sum_b = 0;
		for (int i = 0; i < scores.size(); i++)
		{
			sum_r += src.at<Vec3b>(scores[i].x, scores[i].y).val[2];
			sum_g += src.at<Vec3b>(scores[i].x, scores[i].y).val[1];
			sum_b += src.at<Vec3b>(scores[i].x, scores[i].y).val[0];
		}

		mid_r = sum_r / scores.size();
		mid_g = sum_g / scores.size();
		mid_b = sum_b / scores.size();

	}





};



void Cluster::InitialCenter(int k, Cluster * clusarr, vector<POINT>& vpt, Mat &src) {

	int size = vpt.size();
	int step = size / k;
	int steper = 0;

	for (int i = 0; i < k; i++, steper += step) {
		clusarr[i].curX = vpt[steper + step/2].x;
		clusarr[i].curY = vpt[steper + step / 2].y;

		//Vec3b color = src.at<Vec3b>(m, n);
	//	uchar blue = color.val[0];
	//	uchar green = color.val[1];
	//	uchar red = color.val[2];

		clusarr[i].mid_r = src.at<Vec3b>(clusarr[i].curX, clusarr[i].curY).val[2];
		clusarr[i].mid_g = src.at<Vec3b>(clusarr[i].curX, clusarr[i].curY).val[1];
		clusarr[i].mid_b = src.at<Vec3b>(clusarr[i].curX, clusarr[i].curY).val[0];
	}
}




void Cluster::SetCenter() {
	double sumX = 0, sumY = 0;
	int i = 0;
	int size = Size();
	
	if (size == 0)
	{
	//	active_flag = false;
	//	return;
	}


	for (i; i<size; i++)//the centers of mass by x
	{
		sumX = (sumX * i + scores[i].x) / (i + 1);	
		sumY = (sumY * i + scores[i].y) / (i + 1);
	}


	lastX = curX;
	lastY = curY;

	curX = sumX;
	curY = sumY;

	
}

void Cluster::Clear() {
	scores.clear();
}




double F(double input) // function f(...), which is used for defining L, a and b changes within [4/29,1]
{
	if (input > 0.008856)
		return (pow(input, 0.333333333)); // maximum 1
	else
		return ((841 / 108)*input + 4 / 29);  //841/108 = 29*29/36*16
}

// RGB to XYZ
void RGBtoXYZ(uchar R, uchar G, uchar B, double &X, double &Y, double &Z)
{
	// RGB Working Space: sRGB 
	// Reference White: D65
	X = 0.412453*R + 0.357580*G + 0.189423*B; // maximum value = 0.959456 * 255 = 244.66128
	Y = 0.212671*R + 0.715160*G + 0.072169*B; // maximum value = 1 * 255 = 255
	Z = 0.019334*R + 0.119193*G + 0.950227*B; //  maximum value = 1.088754 * 255 = 277.63227
}

// XYZ to CIELab
void XYZtoLab(double X, double Y, double Z, double &L, double &a, double &b)
{
	const double Xo = 244.66128; // reference white
	const double Yo = 255.0;
	const double Zo = 277.63227;
	L = 116 * F(Y / Yo) - 16; // maximum L = 100
	a = 500 * (F(X / Xo) - F(Y / Yo)); // maximum 
	b = 200 * (F(Y / Yo) - F(Z / Zo));
}

// RGB to CIELab
void RGBtoLab(double R, double G, double B, double &L, double &a, double &b)
{
	double X, Y, Z;
	RGBtoXYZ(R, G, B, X, Y, Z);
	XYZtoLab(X, Y, Z, L, a, b);
}





Cluster * Cluster::Bind(int k, Cluster * clusarr, vector<POINT>& vpt, Mat &src, double c_dig, double d_dig, double c_coef, double d_coef, double max) {
	for (int j = 0; j < k; j++)
		clusarr[j].Clear();// Чистим кластер перед использованием
	int size = vpt.size();
	for (int i = 0; i < size; i++) {// Запускаем цикл по всем пикселям множества
	
		int cur_r = src.at<Vec3b>(vpt[i].x, vpt[i].y).val[2];
		int cur_g = src.at<Vec3b>(vpt[i].x, vpt[i].y).val[1];
		int cur_b = src.at<Vec3b>(vpt[i].x, vpt[i].y).val[0];
		
		//int intensCurrPix = (src.at<Vec3b>(vpt[i].x, vpt[i].y).val[0] + src.at<Vec3b>(vpt[i].x, vpt[i].y).val[1] + src.at<Vec3b>(vpt[i].x, vpt[i].y).val[2]) / 3;

		int col_diff_r = cur_r - clusarr[0].mid_r;
		int col_diff_g = cur_g - clusarr[0].mid_g;
		int col_diff_b = cur_b - clusarr[0].mid_b;

		//int min_col_diff = std::abs(col_diff_r) + std::abs(col_diff_g) + std::abs(col_diff_b);
		// Цветовое расстояние
		//int min_col_diff = sqrt(
		//	pow(cur_r - clusarr[0].mid_r, 2) + pow(cur_g - clusarr[0].mid_g, 2) + pow(cur_r - clusarr[0].mid_b, 2));
		
		
		
		double l, a, b;
		double ml, ma, mb;
		

		RGBtoLab(cur_r, cur_g, cur_b, l, a, b);
		RGBtoLab(clusarr[0].mid_r, clusarr[0].mid_g, clusarr[0].mid_b, ml, ma, mb);

		double min_col_dif_lab = sqrt((l - ml) * (l - ml) + (a - ma)*(a - ma) + (b - mb) * (b - mb));


		
		//----------------TRUE_COLOR_DIST---------------------------\\
		
		int min_col_diff;
		min_col_diff = sqrt((col_diff_b * col_diff_b) + (col_diff_g * col_diff_g) + (col_diff_r * col_diff_r));
		//----------------TRUE_COLOR_DIST---------------------------\\













		//---------------TRUE_GEOM_DIST--------------\\


		int min = sqrt(
			pow((float)clusarr[0].curX - vpt[i].x, 2) + pow((float)clusarr[0].curY - vpt[i].y, 2)
		); // Геометрическое расстояние
		
		   //---------------TRUE_GEOM_DIST--------------\\



		//----------------MATH_PR-------------\\

		min = pow_m(min * d_coef, d_dig, max);
		min_col_dif_lab = pow_m(min_col_dif_lab * c_coef, c_dig, max);


		//----------------MATH_PR---------------//




		double itog = min  + min_col_dif_lab;





		//int intensZeroCentr = (src.at<Vec3b>(clusarr[0].curX, clusarr[0].curY).val[0] + src.at<Vec3b>(clusarr[0].curX, clusarr[0].curY).val[1] +
		//	src.at<Vec3b>(clusarr[0].curX, clusarr[0].curY).val[2]) / 3;





		Cluster * cl = &clusarr[0];
		for (int j = 1; j < k; j++) 
		{
		
			
			//-----------------TRUE_GEOM_DIST------------\\
			
			
			int tmp = sqrt(
				pow((float)clusarr[j].curX - vpt[i].x, 2) + pow((float)clusarr[j].curY - vpt[i].y, 2)
			); 
			
			   //-----------------TRUE_GEOM_DIST------------\\


			int col_diff_r_t = cur_r - clusarr[j].mid_r;
			int col_diff_g_t = cur_g - clusarr[j].mid_g;
			int col_diff_b_t = cur_b - clusarr[j].mid_b;

		//	int min_col_diff_t = sqrt(
		//		pow(cur_r - clusarr[j].mid_r, 2) + pow(cur_g - clusarr[j].mid_g, 2) + pow(cur_r - clusarr[j].mid_b, 2));



			//-------------TRUE_COLOR_DIST----------------\\

			int min_col_diff_t;
			min_col_diff_t = std::abs(col_diff_b_t) + std::abs(col_diff_g_t) + std::abs(col_diff_b_t);






			double lt, at, bt;
			double mlt, mat, mbt;


			RGBtoLab(cur_r, cur_g, cur_b, lt, at, bt);
			RGBtoLab(clusarr[j].mid_r, clusarr[j].mid_g, clusarr[j].mid_b, mlt, mat, mbt);

			double min_col_dif_lab_t = sqrt((lt - mlt) * (lt - mlt) + (at - mat)*(at - mat) + (bt - mbt) * (bt - mbt));









			//-------------TRUE_COLOR_DIST----------------\\


			// Цветовое расстояние с кластером i-м


			//--------------------MATH_PR---------------\\
			
			tmp = pow_m(tmp * d_coef, d_dig, max);
			min_col_dif_lab_t = pow_m(min_col_dif_lab_t * c_coef, c_dig, max);


			//--------------------MATH_PR---------------\\



			double itog_t = tmp + min_col_dif_lab_t;

			
			//int intensOtherCentr = (src.at<Vec3b>(clusarr[j].curX, clusarr[j].curY).val[0] + src.at<Vec3b>(clusarr[j].curX, clusarr[j].curY).val[1] +
			//	src.at<Vec3b>(clusarr[j].curX, clusarr[j].curY).val[2]) / 3;







			if (itog > itog_t) { itog = itog_t; cl = &clusarr[j]; }// Ищем близлежащий кластер
		}
		cl->Add(vpt[i]);// Добавляем в близ лежащий кластер текущий пиксель
	}
	return clusarr;
}








Mat Cluster::Start(int k, Cluster * clusarr, vector<POINT>& vpt, Mat &src, double c_dig, double d_dig, double c_coef, double d_coef, double max) {
	Cluster::InitialCenter(k, clusarr, vpt, src);

	int counter = 0;
	int epselon = 50;
	for (;;) {//Запускаем основной цикл
		int chk = 0;
		Cluster::Bind(k, clusarr, vpt, src, c_dig, d_dig, c_coef, d_coef, max);//Связываем точки с кластерами

		for (int i = 0; i < k; i++)
		clusarr[i].resetMidColor(src); //Пересчитываем средние цвета в кластере

		for (int j = 0; j < k; j++)//Высчитываем новые координаты центроидов 
			clusarr[j].SetCenter();
		for (int p = 0; p < k; p++) {
			
			int ry, rx;

			ry = (clusarr[p].curY - clusarr[p].lastY) * (clusarr[p].curY - clusarr[p].lastY);
			rx = (clusarr[p].curX - clusarr[p].lastX) * (clusarr[p].curX - clusarr[p].lastX);
			//Проверяем не совпадают ли они с предыдущими цент-ми
		//	if (clusarr[p].curX == clusarr[p].lastX && clusarr[p].curY == clusarr[p].lastY)
			//	chk++;
			if (clusarr[p].active_flag)
			{
				if (rx < epselon && ry < epselon) // 4 пикселя не считаем за смещение
					chk++;
			}
			else chk++;


		}





		//----------------ITERATIVE----------------------\\

		/*Mat out_it;
		out_it = src.clone();

		for (int i = 0; i < k; i++) {
			for (int j = 0; j < clusarr[i].scores.size(); j++) {
				int x = clusarr[i].scores[j].x;
				int y = clusarr[i].scores[j].y;

				//	out.at<Vec3b>(x, y).val[2] = clusarr[i].mid_r;
				//	out.at<Vec3b>(x, y).val[1] = clusarr[i].mid_g;
				//	out.at<Vec3b>(x, y).val[0] = clusarr[i].mid_b;

				out_it.at<Vec3b>(x, y) = Vec3b(uchar(50 + 20 * i), uchar(30 + 20 * i), uchar(10 + 20 * i));



			}
		}

		counter++;

		imshow(std::to_string(counter), out_it);

		*/
		//----------------ITERATIVE----------------------\\





		if (chk == k) break;//return;//Если да выходим с цикла
		epselon += 2;
	}


	Mat out;
	out = src.clone();

	for (int i = 0; i < k; i++) {

		
		int r = rand() % 256;
		int g = rand() % 256;
		int b = rand() % 256;

		for (int j = 0; j < clusarr[i].scores.size(); j++) {
			int x = clusarr[i].scores[j].x;
			int y = clusarr[i].scores[j].y;

		//	out.at<Vec3b>(x, y).val[2] = clusarr[i].mid_r;
		//	out.at<Vec3b>(x, y).val[1] = clusarr[i].mid_g;
		//	out.at<Vec3b>(x, y).val[0] = clusarr[i].mid_b;

			


		


			//out.at<Vec3b>(x, y) = Vec3b(uchar(50 + 20 * i), uchar(30 + 20 * i), uchar(10 + 20 * i));
			out.at<Vec3b>(x, y) = Vec3b(uchar(r), uchar(g), uchar(b));


		}
	}
	return out;
}



template <class T>
T pow_m(T ch, double dig, double max)
{
	T out = ch;
	for (int i = 1; i < dig; i++)
	{
		
		if (out > max) return max;
		out *= ch;


	}
	return out;
}