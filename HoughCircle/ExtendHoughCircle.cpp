#include <cv.h>
#include <opencv2\opencv.hpp>
#include <cxmisc.h>
#include "ExtendHoughCircle.h"

#define CV_IMPLEMENT_QSORT_EX( func_name, T, LT, user_data_type )                   \
void func_name( T *array, size_t total, user_data_type aux )                        \
{                                                                                   \
    int isort_thresh = 7;                                                           \
    T t;                                                                            \
    int sp = 0;                                                                     \
                                                                                    \
    struct                                                                          \
    {                                                                               \
        T *lb;                                                                      \
        T *ub;                                                                      \
    }                                                                               \
    stack[48];                                                                      \
                                                                                    \
    (void)aux;                                                                      \
                                                                                    \
    if( total <= 1 )                                                                \
        return;                                                                     \
                                                                                    \
    stack[0].lb = array;                                                            \
    stack[0].ub = array + (total - 1);                                              \
                                                                                    \
    while( sp >= 0 )                                                                \
    {                                                                               \
        T* left = stack[sp].lb;                                                     \
        T* right = stack[sp--].ub;                                                  \
                                                                                    \
        for(;;)                                                                     \
        {                                                                           \
            int i, n = (int)(right - left) + 1, m;                                  \
            T* ptr;                                                                 \
            T* ptr2;                                                                \
                                                                                    \
            if( n <= isort_thresh )                                                 \
            {                                                                       \
            insert_sort:                                                            \
                for( ptr = left + 1; ptr <= right; ptr++ )                          \
                {                                                                   \
                    for( ptr2 = ptr; ptr2 > left && LT(ptr2[0],ptr2[-1]); ptr2--)   \
                        CV_SWAP( ptr2[0], ptr2[-1], t );                            \
                }                                                                   \
                break;                                                              \
            }                                                                       \
            else                                                                    \
            {                                                                       \
                T* left0;                                                           \
                T* left1;                                                           \
                T* right0;                                                          \
                T* right1;                                                          \
                T* pivot;                                                           \
                T* a;                                                               \
                T* b;                                                               \
                T* c;                                                               \
                int swap_cnt = 0;                                                   \
                                                                                    \
                left0 = left;                                                       \
                right0 = right;                                                     \
                pivot = left + (n/2);                                               \
                                                                                    \
                if( n > 40 )                                                        \
                {                                                                   \
                    int d = n / 8;                                                  \
                    a = left, b = left + d, c = left + 2*d;                         \
                    left = LT(*a, *b) ? (LT(*b, *c) ? b : (LT(*a, *c) ? c : a))     \
                                      : (LT(*c, *b) ? b : (LT(*a, *c) ? a : c));    \
                                                                                    \
                    a = pivot - d, b = pivot, c = pivot + d;                        \
                    pivot = LT(*a, *b) ? (LT(*b, *c) ? b : (LT(*a, *c) ? c : a))    \
                                      : (LT(*c, *b) ? b : (LT(*a, *c) ? a : c));    \
                                                                                    \
                    a = right - 2*d, b = right - d, c = right;                      \
                    right = LT(*a, *b) ? (LT(*b, *c) ? b : (LT(*a, *c) ? c : a))    \
                                      : (LT(*c, *b) ? b : (LT(*a, *c) ? a : c));    \
                }                                                                   \
                                                                                    \
                a = left, b = pivot, c = right;                                     \
                pivot = LT(*a, *b) ? (LT(*b, *c) ? b : (LT(*a, *c) ? c : a))        \
                                   : (LT(*c, *b) ? b : (LT(*a, *c) ? a : c));       \
                if( pivot != left0 )                                                \
                {                                                                   \
                    CV_SWAP( *pivot, *left0, t );                                   \
                    pivot = left0;                                                  \
                }                                                                   \
                left = left1 = left0 + 1;                                           \
                right = right1 = right0;                                            \
                                                                                    \
                for(;;)                                                             \
                {                                                                   \
                    while( left <= right && !LT(*pivot, *left) )                    \
                    {                                                               \
                        if( !LT(*left, *pivot) )                                    \
                        {                                                           \
                            if( left > left1 )                                      \
                                CV_SWAP( *left1, *left, t );                        \
                            swap_cnt = 1;                                           \
                            left1++;                                                \
                        }                                                           \
                        left++;                                                     \
                    }                                                               \
                                                                                    \
                    while( left <= right && !LT(*right, *pivot) )                   \
                    {                                                               \
                        if( !LT(*pivot, *right) )                                   \
                        {                                                           \
                            if( right < right1 )                                    \
                                CV_SWAP( *right1, *right, t );                      \
                            swap_cnt = 1;                                           \
                            right1--;                                               \
                        }                                                           \
                        right--;                                                    \
                    }                                                               \
                                                                                    \
                    if( left > right )                                              \
                        break;                                                      \
                    CV_SWAP( *left, *right, t );                                    \
                    swap_cnt = 1;                                                   \
                    left++;                                                         \
                    right--;                                                        \
                }                                                                   \
                                                                                    \
                if( swap_cnt == 0 )                                                 \
                {                                                                   \
                    left = left0, right = right0;                                   \
                    goto insert_sort;                                               \
                }                                                                   \
                                                                                    \
                n = MIN( (int)(left1 - left0), (int)(left - left1) );               \
                for( i = 0; i < n; i++ )                                            \
                    CV_SWAP( left0[i], left[i-n], t );                              \
                                                                                    \
                n = MIN( (int)(right0 - right1), (int)(right1 - right) );           \
                for( i = 0; i < n; i++ )                                            \
                    CV_SWAP( left[i], right0[i-n+1], t );                           \
                n = (int)(left - left1);                                            \
                m = (int)(right1 - right);                                          \
                if( n > 1 )                                                         \
                {                                                                   \
                    if( m > 1 )                                                     \
                    {                                                               \
                        if( n > m )                                                 \
                        {                                                           \
                            stack[++sp].lb = left0;                                 \
                            stack[sp].ub = left0 + n - 1;                           \
                            left = right0 - m + 1, right = right0;                  \
                        }                                                           \
                        else                                                        \
                        {                                                           \
                            stack[++sp].lb = right0 - m + 1;                        \
                            stack[sp].ub = right0;                                  \
                            left = left0, right = left0 + n - 1;                    \
                        }                                                           \
                    }                                                               \
                    else                                                            \
                        left = left0, right = left0 + n - 1;                        \
                }                                                                   \
                else if( m > 1 )                                                    \
                    left = right0 - m + 1, right = right0;                          \
                else                                                                \
                    break;                                                          \
            }                                                                       \
        }                                                                           \
    }                                                                               \
}

#define hough_cmp_gt(l1,l2) (aux[l1] > aux[l2])

static CV_IMPLEMENT_QSORT_EX(icvHoughSortDescent32s, int, hough_cmp_gt, const int*)

static void seqToMat(const CvSeq* seq, cv::OutputArray _arr)
{
	if (seq && seq->total > 0)
	{
		_arr.create(1, seq->total, seq->flags, -1, true);
		cv::Mat arr = _arr.getMat();
		cvCvtSeqToArray(seq, arr.data);
	}
	else
		_arr.release();
}

// 最后确定用分段函数来显示，使暗的更暗，亮的更亮
float fun(float x) {
	float result = 0.0;
	if (x >= 0 && x < HOUGH_MATH_SEGFUN_R1) {
		result = HOUGH_MATH_SEGFUN_S1 / HOUGH_MATH_SEGFUN_R1*x;
	}
	else if (x >= HOUGH_MATH_SEGFUN_R1 && x < HOUGH_MATH_SEGFUN_R2) {
		result = (HOUGH_MATH_SEGFUN_S2 - HOUGH_MATH_SEGFUN_S1) / (HOUGH_MATH_SEGFUN_R2 -
			HOUGH_MATH_SEGFUN_R1)*(x - HOUGH_MATH_SEGFUN_R1) + HOUGH_MATH_SEGFUN_S1;
	}
	else {
		result = (1 - HOUGH_MATH_SEGFUN_S2) / (1 - HOUGH_MATH_SEGFUN_R2)*(x - 1) + 1;
	}
	return result;
}

static void
myicvHoughCirclesGradient(CvMat* img, float dp, float min_dist,
	int min_radius, int max_radius,
	int canny_threshold, int acc_threshold,
	CvSeq* circles, int circles_max)
{
	//为了提高运算精度，定义一个数值的位移量
	const int SHIFT = 10, ONE = 1 << SHIFT;
	//定义水平梯度和垂直梯度矩阵的地址指针
	cv::Ptr<CvMat> dx, dy;
	//定义边缘图像、累加器矩阵和半径距离矩阵的地址指针
	cv::Ptr<CvMat> edges, accum, dist_buf;
	// 创建圆周点梯度方向向量和指向圆心方向向量的内积（小于1）
	cv::Ptr<CvMat> inner_products_buf;
	//定义排序向量
	std::vector<int> sort_buf;
	cv::Ptr<CvMemStorage> storage;

	cv::Ptr<CvMat> dxdy;

	int x, y, i, j, k, center_count, nz_count;
	//事先计算好最小半径和最大半径的平方
	float min_radius2 = (float)min_radius*min_radius;
	float max_radius2 = (float)max_radius*max_radius;
	int rows, cols, arows, acols;
	int astep, *adata;
	float* ddata, *idata;
	//nz表示圆周序列，centers表示圆心序列
	CvSeq *nz, *centers;
	float idp, dr;
	CvSeqReader reader;
	//创建一个边缘图像矩阵
	edges = cvCreateMat(img->rows, img->cols, CV_8UC1);
	//第一阶段
	//步骤1.1，用canny边缘检测算法得到输入图像的边缘图像
	cvCanny(img, edges, MAX(canny_threshold / 2, 1), canny_threshold, 3);
	cvShowImage("edges", edges);
	cvWaitKey(0);
	//创建输入图像的水平梯度图像和垂直梯度图像
	dx = cvCreateMat(img->rows, img->cols, CV_16SC1);
	dy = cvCreateMat(img->rows, img->cols, CV_16SC1);
	//步骤1.2，用Sobel算子法计算水平梯度和垂直梯度
	cvSobel(img, dx, 1, 0, 3);
	cvSobel(img, dy, 0, 1, 3);
	// 确保累加器矩阵的分辨率不小于1
	if (dp < 1.f)
		dp = 1.f;
	//分辨率的倒数
	idp = 1.f / dp;
	//根据分辨率，创建累加器矩阵
	accum = cvCreateMat(cvCeil(img->rows*idp) + 2, cvCeil(img->cols*idp) + 2, CV_32SC1);
	dxdy = cvCreateMat(img->rows, img->cols, CV_16SC2);
	//初始化累加器为0
	cvZero(accum);
	//创建两个序列，
	storage = cvCreateMemStorage();
	nz = cvCreateSeq(CV_32SC2, sizeof(CvSeq), sizeof(CvPoint), storage);
	centers = cvCreateSeq(CV_32SC1, sizeof(CvSeq), sizeof(int), storage);

	rows = img->rows;    //图像的高
	cols = img->cols;    //图像的宽
	arows = accum->rows - 2;    //累加器的高
	acols = accum->cols - 2;    //累加器的宽
	adata = accum->data.i;    //累加器的地址指针
	astep = accum->step / sizeof(adata[0]); // 累加器的步长

	// 便于之后投票值的归一化或者更多的处理，记录accum矩阵中的最大值
	int accum_max = 0;

		// Accumulate circle evidence for each edge pixel
		//步骤1.3，对边缘图像计算累加和
		for (y = 0; y < rows; y++)
		{
			//提取出边缘图像、水平梯度图像和垂直梯度图像的每行的首地址
			const uchar* edges_row = edges->data.ptr + y*edges->step;
			const short* dx_row = (const short*)(dx->data.ptr + y*dx->step);
			const short* dy_row = (const short*)(dy->data.ptr + y*dy->step);
			short* dxdy_row = (short*)(dxdy->data.ptr + y*dxdy->step);

			for (x = 0; x < cols; x++)
			{
				float vx, vy;
				int sx, sy, x0, y0, x1, y1, r;
				CvPoint pt;
				//当前的水平梯度值和垂直梯度值
				vx = dx_row[x];
				vy = dy_row[x];
				//如果当前的像素不是边缘点，或者水平梯度值和垂直梯度值都为0，则继续循环。因为如果满足上面条件，该点一定不是圆周上的点
				if (!edges_row[x] || (vx == 0 && vy == 0))
					continue;
				//计算当前点的梯度值
				float mag = sqrt(vx*vx + vy*vy);
				assert(mag >= 1);
				//定义水平和垂直的位移量
				sx = cvRound((vx*idp)*ONE / mag);
				sy = cvRound((vy*idp)*ONE / mag);

				// 梯度向量
				dxdy_row[2 * x + 0] = sx;
				dxdy_row[2 * x + 1] = sy;

				//把当前点的坐标定位到累加器的位置上
				x0 = cvRound((x*idp)*ONE);
				y0 = cvRound((y*idp)*ONE);
				// Step from min_radius to max_radius in both directions of the gradient
				//在梯度的两个方向上进行位移，并对累加器进行投票累计
				for (int k1 = 0; k1 < 2; k1++)
				{
					//初始一个位移的启动
					//位移量乘以最小半径，从而保证了所检测的圆的半径一定是大于最小半径
					x1 = x0 + min_radius * sx;
					y1 = y0 + min_radius * sy;
					//在梯度的方向上位移
					// r <= max_radius保证了所检测的圆的半径一定是小于最大半径
					for (r = min_radius; r <= max_radius; x1 += sx, y1 += sy, r++)
					{
						int x2 = x1 >> SHIFT, y2 = y1 >> SHIFT;
						//如果位移后的点超过了累加器矩阵的范围，则退出
						if ((unsigned)x2 >= (unsigned)acols ||
							(unsigned)y2 >= (unsigned)arows)
							break;
						//在累加器的相应位置上加1
						adata[y2*astep + x2]++;
						if (adata[y2*astep + x2] > accum_max) {
							accum_max = adata[y2*astep + x2];
						}
					}
					//把位移量设置为反方向
					sx = -sx; sy = -sy;
				}
				//把输入图像中的当前点（即圆周上的点）的坐标压入序列圆周序列nz中
				pt.x = x; pt.y = y;
				cvSeqPush(nz, &pt);
			}
		}
	//计算圆周点的总数
	nz_count = nz->total;
	//如果总数为0，说明没有检测到圆，则退出该函数
	if (!nz_count)
		return;
	//Find possible circle centers
	//步骤1.4和1.5，遍历整个累加器矩阵，找到可能的圆心

	// 重新对累加器中的值归一化或者进行更多的处理
	for (y = 1; y < arows - 1; y++)
	{
		for (x = 1; x < acols - 1; x++)
		{
			int base = y*(acols + 2) + x;
			//adata[base] = ((float)adata[base] / accum_max) * HOUGH_CIRCLE_ACCUM_NORMALIZE_MAX;
			adata[base] = fun((float)adata[base] / accum_max) * HOUGH_CIRCLE_ACCUM_NORMALIZE_MAX;
		}
	}


	for (y = 1; y < arows - 1; y++)
	{
		for (x = 1; x < acols - 1; x++)
		{
			int base = y*(acols + 2) + x;
			//如果当前的值大于阈值，并在4邻域内它是最大值，则该点被认为是圆心
			if (adata[base] > acc_threshold &&
				adata[base] > adata[base - 1] && adata[base] > adata[base + 1] &&
				adata[base] > adata[base - acols - 2] && adata[base] > adata[base + acols + 2])
				//把当前点的地址压入圆心序列centers中
				cvSeqPush(centers, &base);
		}
	}

	// 显示一下accumaltor中的投票数
	cv::Ptr<CvMat> acc_img;
	acc_img = cvCreateMat(accum->rows, accum->cols, CV_8UC1);
	cvNormalize(accum, acc_img, 0, 255, cv::NORM_MINMAX);
	cvShowImage("accum", acc_img);
	cvWaitKey(0);

	//计算圆心的总数
	center_count = centers->total;
	//如果总数为0，说明没有检测到圆，则退出该函数
	if (!center_count)
		return;
	//定义排序向量的大小
	sort_buf.resize(MAX(center_count, nz_count));
	//把圆心序列放入排序向量中
	cvCvtSeqToArray(centers, &sort_buf[0]);
	//对圆心按照由大到小的顺序进行排序
	//它的原理是经过icvHoughSortDescent32s函数后，以sort_buf中元素作为adata数组下标，adata中的元素降序排列，即adata[sort_buf[0]]是adata所有元素中最大的，adata[sort_buf[center_count-1]]是所有元素中最小的
	icvHoughSortDescent32s(&sort_buf[0], center_count, adata);
	//清空圆心序列
	cvClearSeq(centers);
	//把排好序的圆心重新放入圆心序列中
	cvSeqPushMulti(centers, &sort_buf[0], center_count);
	//创建半径距离矩阵
	dist_buf = cvCreateMat(1, nz_count, CV_32FC1);
	//定义地址指针
	ddata = dist_buf->data.fl;

	inner_products_buf = cvCreateMat(1, nz_count, CV_32FC1);
	//定义地址指针
	idata = inner_products_buf->data.fl;


	dr = dp;    //定义圆半径的距离分辨率
				//重新定义圆心之间的最小距离
	min_dist = MAX(min_dist, dp);
	//最小距离的平方
	min_dist *= min_dist;
	// For each found possible center
	// Estimate radius and check support
	//按照由大到小的顺序遍历整个圆心序列
	for (i = 0; i < centers->total; i++)
	{
		//提取出圆心，得到该点在累加器矩阵中的偏移量
		int ofs = *(int*)cvGetSeqElem(centers, i);
		//得到圆心在累加器中的坐标位置
		y = ofs / (acols + 2);
		x = ofs - (y)*(acols + 2);
		//Calculate circle's center in pixels
		//计算圆心在输入图像中的坐标位置
		float cx = (float)((x + 0.5f)*dp), cy = (float)((y + 0.5f)*dp);
		float start_dist, dist_sum;
		float r_best = 0;
		int max_count = 0;
		// Check distance with previously detected circles
		//判断当前的圆心与之前确定作为输出的圆心是否为同一个圆心
		for (j = 0; j < circles->total; j++)
		{
			//从序列中提取出圆心
			float* c = (float*)cvGetSeqElem(circles, j);
			//计算当前圆心与提取出的圆心之间的距离，如果两者距离小于所设的阈值，则认为两个圆心是同一个圆心，退出循环
			if ((c[0] - cx)*(c[0] - cx) + (c[1] - cy)*(c[1] - cy) < min_dist)
				break;
		}
		//如果j < circles->total，说明当前的圆心已被认为与之前确定作为输出的圆心是同一个圆心，则抛弃该圆心，返回上面的for循环
		if (j < circles->total)
			continue;
		// Estimate best radius
		//第二阶段
		//开始读取圆周序列nz
		cvStartReadSeq(nz, &reader);
		for (j = k = 0; j < nz_count; j++)
		{
			CvPoint pt;

			float _dx, _dy, _r2;
			CV_READ_SEQ_ELEM(pt, reader);
			_dx = cx - pt.x; _dy = cy - pt.y;
			//步骤2.1，计算圆周上的点与当前圆心的距离，即半径
			_r2 = _dx*_dx + _dy*_dy;

			float x_norm = _dx / pow(_r2, 0.5);
			float y_norm = _dy / pow(_r2, 0.5);

			// 求得sx,dx（记得求完内积后右移SHIFT位）
			const short* dxdy_row = (short*)(dxdy->data.ptr + pt.y*dxdy->step);
			short sx = dxdy_row[2 * pt.x + 0];
			short sy = dxdy_row[2 * pt.x + 1];

			//步骤2.2，如果半径在所设置的最大半径和最小半径之间
			if (min_radius2 <= _r2 && _r2 <= max_radius2)
			{
				//把半径存入dist_buf内
				ddata[k] = _r2;
				sort_buf[k] = k;
				idata[k] = sx * x_norm + sy * y_norm;
				k++;
			}
		}
		//k表示一共有多少个圆周上的点
		int nz_count1 = k, start_idx = nz_count1 - 1;
		//nz_count1等于0也就是k等于0，说明当前的圆心没有所对应的圆，意味着当前圆心不是真正的圆心，所以抛弃该圆心，返回上面的for循环
		if (nz_count1 == 0)
			continue;
		dist_buf->cols = nz_count1;    //得到圆周上点的个数
		cvPow(dist_buf, dist_buf, 0.5);    //求平方根，得到真正的圆半径
										   //步骤2.3，对圆半径进行排序
		icvHoughSortDescent32s(&sort_buf[0], nz_count1, (int*)ddata);

		dist_sum = start_dist = ddata[sort_buf[nz_count1 - 1]];
		float cur_r_dist_sum = 0.0;
		int cur_r_count = 0;
		int cur_r_grad_count = 0;
		//步骤2.4
		for (j = nz_count1 - 2; j >= 0; j--)
		{
			float d = ddata[sort_buf[j]];
			float inner_product = idata[sort_buf[j]];

			if (d > max_radius)
				break;
			//d表示当前半径值，start_dist表示上一次通过下面if语句更新后的半径值，dr表示半径距离分辨率，如果这两个半径距离之差大于距离分辨率，说明这两个半径一定不属于同一个圆，而两次满足if语句条件之间的那些半径值可以认为是相等的，即是属于同一个圆
			if (d - start_dist < HOUGH_CIRCLE_RADIUS_MIN_DIST * dr)
			{
				////start_idx表示上一次进入if语句时更新的半径距离排序的序号
				//// start_idx – j表示当前得到的相同半径距离的数量
				////(j + start_idx)/2表示j和start_idx中间的数
				////取中间的数所对应的半径值作为当前半径值r_cur，也就是取那些半径值相同的值
				//float r_cur = ddata[sort_buf[(j + start_idx) / 2]];
				////如果当前得到的半径相同的数量大于最大值max_count，则进入if语句
				//if ((start_idx - j)*r_best >= max_count*r_cur ||
				//	(r_best < FLT_EPSILON && start_idx - j >= max_count))
				//{
				//	r_best = r_cur;    //把当前半径值作为最佳半径值
				//	max_count = start_idx - j;    //更新最大值
				//}
				////更新半径距离和序号
				//start_dist = d;
				//start_idx = j;
				//dist_sum = 0;

				// 重新修改代码，使得检测过程中单圆心可以对应多个半径，并且尽量减少该算法对参数的依赖
				cur_r_count++;
				cur_r_dist_sum += d;
				// 增加梯度检测
				if (fabs(inner_product) > HOUGH_CIRCLE_SAMEDIRECT_DEGREE * ONE) {
					cur_r_grad_count++;
				}
			}
			// 说明此时已经是另一个圆，因此需要判定该圆是否合格，并且要更新某些局部变量，为了统计下一个圆
			else {
				// 计算平均半径
				float r_mean = cur_r_dist_sum / cur_r_count;
				// 判断该圆是否合格
				if (cur_r_count >= HOUGH_CIRCLE_INTEGRITY_DEGREE * 2 * HOUGH_MATH_PI * r_mean &&
					cur_r_grad_count >= HOUGH_CIRCLE_GRADIENT_INTEGRITY_DEGREE * cur_r_count){
					float c[3];
					c[0] = cx;    //圆心的横坐标
					c[1] = cy;    //圆心的纵坐标
					c[2] = (float)r_mean;    //所对应的圆的半径
					cvSeqPush(circles, c);    //压入序列circles内
											  //如果得到的圆大于阈值，则退出该函数
					if (circles->total > circles_max)
						return;
				}
				cur_r_count = 1;
				cur_r_dist_sum = d;
				start_dist = d;
			}
		}
	}
}

static void
icvHoughCirclesGradient(CvMat* img, float dp, float min_dist,
	int min_radius, int max_radius,
	int canny_threshold, int acc_threshold,
	CvSeq* circles, int circles_max)
{
	//为了提高运算精度，定义一个数值的位移量
	const int SHIFT = 10, ONE = 1 << SHIFT;
	//定义水平梯度和垂直梯度矩阵的地址指针
	cv::Ptr<CvMat> dx, dy;
	//定义边缘图像、累加器矩阵和半径距离矩阵的地址指针
	cv::Ptr<CvMat> edges, accum, dist_buf;
	//定义排序向量
	std::vector<int> sort_buf;
	cv::Ptr<CvMemStorage> storage;

	int x, y, i, j, k, center_count, nz_count;
	//事先计算好最小半径和最大半径的平方
	float min_radius2 = (float)min_radius*min_radius;
	float max_radius2 = (float)max_radius*max_radius;
	int rows, cols, arows, acols;
	int astep, *adata;
	float* ddata;
	//nz表示圆周序列，centers表示圆心序列
	CvSeq *nz, *centers;
	float idp, dr;
	CvSeqReader reader;
	//创建一个边缘图像矩阵
	edges = cvCreateMat(img->rows, img->cols, CV_8UC1);
	//第一阶段
	//步骤1.1，用canny边缘检测算法得到输入图像的边缘图像
	cvCanny(img, edges, MAX(canny_threshold / 2, 1), canny_threshold, 3);
	cvShowImage("edges", edges);
	cvWaitKey(0);
	//创建输入图像的水平梯度图像和垂直梯度图像
	dx = cvCreateMat(img->rows, img->cols, CV_16SC1);
	dy = cvCreateMat(img->rows, img->cols, CV_16SC1);
	//步骤1.2，用Sobel算子法计算水平梯度和垂直梯度
	cvSobel(img, dx, 1, 0, 3);
	cvSobel(img, dy, 0, 1, 3);
	// 确保累加器矩阵的分辨率不小于1
	if (dp < 1.f)
		dp = 1.f;
	//分辨率的倒数
	idp = 1.f / dp;
	//根据分辨率，创建累加器矩阵
	accum = cvCreateMat(cvCeil(img->rows*idp) + 2, cvCeil(img->cols*idp) + 2, CV_32SC1);
	//初始化累加器为0
	cvZero(accum);
	//创建两个序列，
	storage = cvCreateMemStorage();
	nz = cvCreateSeq(CV_32SC2, sizeof(CvSeq), sizeof(CvPoint), storage);
	centers = cvCreateSeq(CV_32SC1, sizeof(CvSeq), sizeof(int), storage);

	rows = img->rows;    //图像的高
	cols = img->cols;    //图像的宽
	arows = accum->rows - 2;    //累加器的高
	acols = accum->cols - 2;    //累加器的宽
	adata = accum->data.i;    //累加器的地址指针
	astep = accum->step / sizeof(adata[0]); // 累加器的步长
											// Accumulate circle evidence for each edge pixel
											//步骤1.3，对边缘图像计算累加和
	for (y = 0; y < rows; y++)
	{
		//提取出边缘图像、水平梯度图像和垂直梯度图像的每行的首地址
		const uchar* edges_row = edges->data.ptr + y*edges->step;
		const short* dx_row = (const short*)(dx->data.ptr + y*dx->step);
		const short* dy_row = (const short*)(dy->data.ptr + y*dy->step);

		for (x = 0; x < cols; x++)
		{
			float vx, vy;
			int sx, sy, x0, y0, x1, y1, r;
			CvPoint pt;
			//当前的水平梯度值和垂直梯度值
			vx = dx_row[x];
			vy = dy_row[x];
			//如果当前的像素不是边缘点，或者水平梯度值和垂直梯度值都为0，则继续循环。因为如果满足上面条件，该点一定不是圆周上的点
			if (!edges_row[x] || (vx == 0 && vy == 0))
				continue;
			//计算当前点的梯度值
			float mag = sqrt(vx*vx + vy*vy);
			assert(mag >= 1);
			//定义水平和垂直的位移量
			sx = cvRound((vx*idp)*ONE / mag);
			sy = cvRound((vy*idp)*ONE / mag);
			//把当前点的坐标定位到累加器的位置上
			x0 = cvRound((x*idp)*ONE);
			y0 = cvRound((y*idp)*ONE);
			// Step from min_radius to max_radius in both directions of the gradient
			//在梯度的两个方向上进行位移，并对累加器进行投票累计
			for (int k1 = 0; k1 < 2; k1++)
			{
				//初始一个位移的启动
				//位移量乘以最小半径，从而保证了所检测的圆的半径一定是大于最小半径
				x1 = x0 + min_radius * sx;
				y1 = y0 + min_radius * sy;
				//在梯度的方向上位移
				// r <= max_radius保证了所检测的圆的半径一定是小于最大半径
				for (r = min_radius; r <= max_radius; x1 += sx, y1 += sy, r++)
				{
					int x2 = x1 >> SHIFT, y2 = y1 >> SHIFT;
					//如果位移后的点超过了累加器矩阵的范围，则退出
					if ((unsigned)x2 >= (unsigned)acols ||
						(unsigned)y2 >= (unsigned)arows)
						break;
					//在累加器的相应位置上加1
					adata[y2*astep + x2]++;
				}
				//把位移量设置为反方向
				sx = -sx; sy = -sy;
			}
			//把输入图像中的当前点（即圆周上的点）的坐标压入序列圆周序列nz中
			pt.x = x; pt.y = y;
			cvSeqPush(nz, &pt);
		}
	}
	//计算圆周点的总数
	nz_count = nz->total;
	//如果总数为0，说明没有检测到圆，则退出该函数
	if (!nz_count)
		return;
	//Find possible circle centers
	//步骤1.4和1.5，遍历整个累加器矩阵，找到可能的圆心
	for (y = 1; y < arows - 1; y++)
	{
		for (x = 1; x < acols - 1; x++)
		{
			int base = y*(acols + 2) + x;
			//如果当前的值大于阈值，并在4邻域内它是最大值，则该点被认为是圆心
			if (adata[base] > acc_threshold &&
				adata[base] > adata[base - 1] && adata[base] > adata[base + 1] &&
				adata[base] > adata[base - acols - 2] && adata[base] > adata[base + acols + 2])
				//把当前点的地址压入圆心序列centers中
				cvSeqPush(centers, &base);
		}
	}
	//计算圆心的总数
	center_count = centers->total;
	//如果总数为0，说明没有检测到圆，则退出该函数
	if (!center_count)
		return;
	//定义排序向量的大小
	sort_buf.resize(MAX(center_count, nz_count));
	//把圆心序列放入排序向量中
	cvCvtSeqToArray(centers, &sort_buf[0]);
	//对圆心按照由大到小的顺序进行排序
	//它的原理是经过icvHoughSortDescent32s函数后，以sort_buf中元素作为adata数组下标，adata中的元素降序排列，即adata[sort_buf[0]]是adata所有元素中最大的，adata[sort_buf[center_count-1]]是所有元素中最小的
	icvHoughSortDescent32s(&sort_buf[0], center_count, adata);
	//清空圆心序列
	cvClearSeq(centers);
	//把排好序的圆心重新放入圆心序列中
	cvSeqPushMulti(centers, &sort_buf[0], center_count);
	//创建半径距离矩阵
	dist_buf = cvCreateMat(1, nz_count, CV_32FC1);
	//定义地址指针
	ddata = dist_buf->data.fl;

	dr = dp;    //定义圆半径的距离分辨率
				//重新定义圆心之间的最小距离
	min_dist = MAX(min_dist, dp);
	//最小距离的平方
	min_dist *= min_dist;
	// For each found possible center
	// Estimate radius and check support
	//按照由大到小的顺序遍历整个圆心序列
	for (i = 0; i < centers->total; i++)
	{
		//提取出圆心，得到该点在累加器矩阵中的偏移量
		int ofs = *(int*)cvGetSeqElem(centers, i);
		//得到圆心在累加器中的坐标位置
		y = ofs / (acols + 2);
		x = ofs - (y)*(acols + 2);
		//Calculate circle's center in pixels
		//计算圆心在输入图像中的坐标位置
		float cx = (float)((x + 0.5f)*dp), cy = (float)((y + 0.5f)*dp);
		float start_dist, dist_sum;
		float r_best = 0;
		int max_count = 0;
		// Check distance with previously detected circles
		//判断当前的圆心与之前确定作为输出的圆心是否为同一个圆心
		for (j = 0; j < circles->total; j++)
		{
			//从序列中提取出圆心
			float* c = (float*)cvGetSeqElem(circles, j);
			//计算当前圆心与提取出的圆心之间的距离，如果两者距离小于所设的阈值，则认为两个圆心是同一个圆心，退出循环
			if ((c[0] - cx)*(c[0] - cx) + (c[1] - cy)*(c[1] - cy) < min_dist)
				break;
		}
		//如果j < circles->total，说明当前的圆心已被认为与之前确定作为输出的圆心是同一个圆心，则抛弃该圆心，返回上面的for循环
		if (j < circles->total)
			continue;
		// Estimate best radius
		//第二阶段
		//开始读取圆周序列nz
		cvStartReadSeq(nz, &reader);
		for (j = k = 0; j < nz_count; j++)
		{
			CvPoint pt;
			float _dx, _dy, _r2;
			CV_READ_SEQ_ELEM(pt, reader);
			_dx = cx - pt.x; _dy = cy - pt.y;
			//步骤2.1，计算圆周上的点与当前圆心的距离，即半径
			_r2 = _dx*_dx + _dy*_dy;
			//步骤2.2，如果半径在所设置的最大半径和最小半径之间
			if (min_radius2 <= _r2 && _r2 <= max_radius2)
			{
				//把半径存入dist_buf内
				ddata[k] = _r2;
				sort_buf[k] = k;
				k++;
			}
		}
		//k表示一共有多少个圆周上的点
		int nz_count1 = k, start_idx = nz_count1 - 1;
		//nz_count1等于0也就是k等于0，说明当前的圆心没有所对应的圆，意味着当前圆心不是真正的圆心，所以抛弃该圆心，返回上面的for循环
		if (nz_count1 == 0)
			continue;
		dist_buf->cols = nz_count1;    //得到圆周上点的个数
		cvPow(dist_buf, dist_buf, 0.5);    //求平方根，得到真正的圆半径
										   //步骤2.3，对圆半径进行排序
		icvHoughSortDescent32s(&sort_buf[0], nz_count1, (int*)ddata);

		dist_sum = start_dist = ddata[sort_buf[nz_count1 - 1]];
		//步骤2.4
		for (j = nz_count1 - 2; j >= 0; j--)
		{
			float d = ddata[sort_buf[j]];

			if (d > max_radius)
				break;
			//d表示当前半径值，start_dist表示上一次通过下面if语句更新后的半径值，dr表示半径距离分辨率，如果这两个半径距离之差大于距离分辨率，说明这两个半径一定不属于同一个圆，而两次满足if语句条件之间的那些半径值可以认为是相等的，即是属于同一个圆
			if (d - start_dist > dr)
			{
				//start_idx表示上一次进入if语句时更新的半径距离排序的序号
				// start_idx – j表示当前得到的相同半径距离的数量
				//(j + start_idx)/2表示j和start_idx中间的数
				//取中间的数所对应的半径值作为当前半径值r_cur，也就是取那些半径值相同的值
				float r_cur = ddata[sort_buf[(j + start_idx) / 2]];
				//如果当前得到的半径相同的数量大于最大值max_count，则进入if语句
				if ((start_idx - j)*r_best >= max_count*r_cur ||
					(r_best < FLT_EPSILON && start_idx - j >= max_count))
				{
					r_best = r_cur;    //把当前半径值作为最佳半径值
					max_count = start_idx - j;    //更新最大值
				}
				//更新半径距离和序号
				start_dist = d;
				start_idx = j;
				dist_sum = 0;
			}
			dist_sum += d;
		}
		// Check if the circle has enough support
		//步骤2.5，最终确定输出
		//如果相同半径的数量大于所设阈值
		if (max_count > acc_threshold)
		{
			float c[3];
			c[0] = cx;    //圆心的横坐标
			c[1] = cy;    //圆心的纵坐标
			c[2] = (float)r_best;    //所对应的圆的半径
			cvSeqPush(circles, c);    //压入序列circles内
									  //如果得到的圆大于阈值，则退出该函数
			if (circles->total > circles_max)
				return;
		}
	}
}

CV_IMPL CvSeq*
cvHoughCircles(CvArr* src_image, void* circle_storage,
	int method, double dp, double min_dist,
	double param1, double param2,
	int min_radius, int max_radius)
{
	CvSeq* result = 0;

	CvMat stub, *img = (CvMat*)src_image;
	CvMat* mat = 0;
	CvSeq* circles = 0;
	CvSeq circles_header;
	CvSeqBlock circles_block;
	int circles_max = INT_MAX;    //输出最多圆形的数量，设为无穷多
	//canny边缘检测中双阈值中的高阈值
	int canny_threshold = cvRound(param1);
	//累加器阈值
	int acc_threshold = cvRound(param2);

	img = cvGetMat(img, &stub);
	//确保输入图像是灰度图像
	if (!CV_IS_MASK_ARR(img))
		CV_Error(CV_StsBadArg, "The source image must be 8-bit, single-channel");
	//内存空间是否存在
	if (!circle_storage)
		CV_Error(CV_StsNullPtr, "NULL destination");
	//确保参数的正确性
	if (dp <= 0 || min_dist <= 0 || canny_threshold <= 0 || acc_threshold <= 0)
		CV_Error(CV_StsOutOfRange, "dp, min_dist, canny_threshold and acc_threshold must be all positive numbers");
	//圆的最小半径要大于0
	min_radius = MAX(min_radius, HOUGH_CIRCLE_RADIUS_MIN);
	//圆的最大半径如果小于等于0，则设最大半径为图像宽和长度的最大值，
	//如果最大半径小于最小半径，则设最大半径为最小半径加两个像素的宽度
	if (max_radius <= 0)
		max_radius = MAX(img->rows, img->cols);
	else if (max_radius <= min_radius)
		max_radius = min_radius + 2;

	if (CV_IS_STORAGE(circle_storage))
	{
		circles = cvCreateSeq(CV_32FC3, sizeof(CvSeq),
			sizeof(float) * 3, (CvMemStorage*)circle_storage);
	}
	else if (CV_IS_MAT(circle_storage))
	{
		mat = (CvMat*)circle_storage;

		if (!CV_IS_MAT_CONT(mat->type) || (mat->rows != 1 && mat->cols != 1) ||
			CV_MAT_TYPE(mat->type) != CV_32FC3)
			CV_Error(CV_StsBadArg,
				"The destination matrix should be continuous and have a single row or a single column");

		circles = cvMakeSeqHeaderForArray(CV_32FC3, sizeof(CvSeq), sizeof(float) * 3,
			mat->data.ptr, mat->rows + mat->cols - 1, &circles_header, &circles_block);
		circles_max = circles->total;
		cvClearSeq(circles);
	}
	else
		CV_Error(CV_StsBadArg, "Destination is not CvMemStorage* nor CvMat*");
	//选择哪种算法检测圆，目前只有2-1霍夫变换
	switch (method)
	{
	case CV_HOUGH_GRADIENT:
		//调用icvHoughCirclesGradient函数
		myicvHoughCirclesGradient(img, (float)dp, (float)min_dist,
			min_radius, max_radius, canny_threshold,
			acc_threshold, circles, circles_max);
		break;
	default:
		CV_Error(CV_StsBadArg, "Unrecognized method id");
	}

	if (mat)
	{
		if (mat->cols > mat->rows)
			mat->cols = circles->total;
		else
			mat->rows = circles->total;
	}
	else
		result = circles;
	//输出圆
	return result;
}


void myHoughCircles( cv::InputArray _image, cv::OutputArray _circles,
int method, double dp, double min_dist,
double param1, double param2,
int minRadius, int maxRadius )
{
	//定义一段内存
	cv::Ptr<CvMemStorage> storage = cvCreateMemStorage(0);
	cv::Mat image = _image.getMat();    //提取输入图像矩阵
	CvMat c_image = image;    //矩阵转换
							  //调用cvHoughCircles函数
	CvSeq* seq = cvHoughCircles(&c_image, storage, method,
		dp, min_dist, param1, param2, minRadius, maxRadius);
	//把序列转换为矩阵
	seqToMat(seq, _circles);
}
