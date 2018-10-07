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

// ���ȷ���÷ֶκ�������ʾ��ʹ���ĸ��������ĸ���
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
	//Ϊ��������㾫�ȣ�����һ����ֵ��λ����
	const int SHIFT = 10, ONE = 1 << SHIFT;
	//����ˮƽ�ݶȺʹ�ֱ�ݶȾ���ĵ�ַָ��
	cv::Ptr<CvMat> dx, dy;
	//�����Եͼ���ۼ�������Ͱ뾶�������ĵ�ַָ��
	cv::Ptr<CvMat> edges, accum, dist_buf;
	// ����Բ�ܵ��ݶȷ���������ָ��Բ�ķ����������ڻ���С��1��
	cv::Ptr<CvMat> inner_products_buf;
	//������������
	std::vector<int> sort_buf;
	cv::Ptr<CvMemStorage> storage;

	cv::Ptr<CvMat> dxdy;

	int x, y, i, j, k, center_count, nz_count;
	//���ȼ������С�뾶�����뾶��ƽ��
	float min_radius2 = (float)min_radius*min_radius;
	float max_radius2 = (float)max_radius*max_radius;
	int rows, cols, arows, acols;
	int astep, *adata;
	float* ddata, *idata;
	//nz��ʾԲ�����У�centers��ʾԲ������
	CvSeq *nz, *centers;
	float idp, dr;
	CvSeqReader reader;
	//����һ����Եͼ�����
	edges = cvCreateMat(img->rows, img->cols, CV_8UC1);
	//��һ�׶�
	//����1.1����canny��Ե����㷨�õ�����ͼ��ı�Եͼ��
	cvCanny(img, edges, MAX(canny_threshold / 2, 1), canny_threshold, 3);
	cvShowImage("edges", edges);
	cvWaitKey(0);
	//��������ͼ���ˮƽ�ݶ�ͼ��ʹ�ֱ�ݶ�ͼ��
	dx = cvCreateMat(img->rows, img->cols, CV_16SC1);
	dy = cvCreateMat(img->rows, img->cols, CV_16SC1);
	//����1.2����Sobel���ӷ�����ˮƽ�ݶȺʹ�ֱ�ݶ�
	cvSobel(img, dx, 1, 0, 3);
	cvSobel(img, dy, 0, 1, 3);
	// ȷ���ۼ�������ķֱ��ʲ�С��1
	if (dp < 1.f)
		dp = 1.f;
	//�ֱ��ʵĵ���
	idp = 1.f / dp;
	//���ݷֱ��ʣ������ۼ�������
	accum = cvCreateMat(cvCeil(img->rows*idp) + 2, cvCeil(img->cols*idp) + 2, CV_32SC1);
	dxdy = cvCreateMat(img->rows, img->cols, CV_16SC2);
	//��ʼ���ۼ���Ϊ0
	cvZero(accum);
	//�����������У�
	storage = cvCreateMemStorage();
	nz = cvCreateSeq(CV_32SC2, sizeof(CvSeq), sizeof(CvPoint), storage);
	centers = cvCreateSeq(CV_32SC1, sizeof(CvSeq), sizeof(int), storage);

	rows = img->rows;    //ͼ��ĸ�
	cols = img->cols;    //ͼ��Ŀ�
	arows = accum->rows - 2;    //�ۼ����ĸ�
	acols = accum->cols - 2;    //�ۼ����Ŀ�
	adata = accum->data.i;    //�ۼ����ĵ�ַָ��
	astep = accum->step / sizeof(adata[0]); // �ۼ����Ĳ���

	// ����֮��ͶƱֵ�Ĺ�һ�����߸���Ĵ�����¼accum�����е����ֵ
	int accum_max = 0;

		// Accumulate circle evidence for each edge pixel
		//����1.3���Ա�Եͼ������ۼӺ�
		for (y = 0; y < rows; y++)
		{
			//��ȡ����Եͼ��ˮƽ�ݶ�ͼ��ʹ�ֱ�ݶ�ͼ���ÿ�е��׵�ַ
			const uchar* edges_row = edges->data.ptr + y*edges->step;
			const short* dx_row = (const short*)(dx->data.ptr + y*dx->step);
			const short* dy_row = (const short*)(dy->data.ptr + y*dy->step);
			short* dxdy_row = (short*)(dxdy->data.ptr + y*dxdy->step);

			for (x = 0; x < cols; x++)
			{
				float vx, vy;
				int sx, sy, x0, y0, x1, y1, r;
				CvPoint pt;
				//��ǰ��ˮƽ�ݶ�ֵ�ʹ�ֱ�ݶ�ֵ
				vx = dx_row[x];
				vy = dy_row[x];
				//�����ǰ�����ز��Ǳ�Ե�㣬����ˮƽ�ݶ�ֵ�ʹ�ֱ�ݶ�ֵ��Ϊ0�������ѭ������Ϊ������������������õ�һ������Բ���ϵĵ�
				if (!edges_row[x] || (vx == 0 && vy == 0))
					continue;
				//���㵱ǰ����ݶ�ֵ
				float mag = sqrt(vx*vx + vy*vy);
				assert(mag >= 1);
				//����ˮƽ�ʹ�ֱ��λ����
				sx = cvRound((vx*idp)*ONE / mag);
				sy = cvRound((vy*idp)*ONE / mag);

				// �ݶ�����
				dxdy_row[2 * x + 0] = sx;
				dxdy_row[2 * x + 1] = sy;

				//�ѵ�ǰ������궨λ���ۼ�����λ����
				x0 = cvRound((x*idp)*ONE);
				y0 = cvRound((y*idp)*ONE);
				// Step from min_radius to max_radius in both directions of the gradient
				//���ݶȵ����������Ͻ���λ�ƣ������ۼ�������ͶƱ�ۼ�
				for (int k1 = 0; k1 < 2; k1++)
				{
					//��ʼһ��λ�Ƶ�����
					//λ����������С�뾶���Ӷ���֤��������Բ�İ뾶һ���Ǵ�����С�뾶
					x1 = x0 + min_radius * sx;
					y1 = y0 + min_radius * sy;
					//���ݶȵķ�����λ��
					// r <= max_radius��֤��������Բ�İ뾶һ����С�����뾶
					for (r = min_radius; r <= max_radius; x1 += sx, y1 += sy, r++)
					{
						int x2 = x1 >> SHIFT, y2 = y1 >> SHIFT;
						//���λ�ƺ�ĵ㳬�����ۼ�������ķ�Χ�����˳�
						if ((unsigned)x2 >= (unsigned)acols ||
							(unsigned)y2 >= (unsigned)arows)
							break;
						//���ۼ�������Ӧλ���ϼ�1
						adata[y2*astep + x2]++;
						if (adata[y2*astep + x2] > accum_max) {
							accum_max = adata[y2*astep + x2];
						}
					}
					//��λ��������Ϊ������
					sx = -sx; sy = -sy;
				}
				//������ͼ���еĵ�ǰ�㣨��Բ���ϵĵ㣩������ѹ������Բ������nz��
				pt.x = x; pt.y = y;
				cvSeqPush(nz, &pt);
			}
		}
	//����Բ�ܵ������
	nz_count = nz->total;
	//�������Ϊ0��˵��û�м�⵽Բ�����˳��ú���
	if (!nz_count)
		return;
	//Find possible circle centers
	//����1.4��1.5�����������ۼ��������ҵ����ܵ�Բ��

	// ���¶��ۼ����е�ֵ��һ�����߽��и���Ĵ���
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
			//�����ǰ��ֵ������ֵ������4�������������ֵ����õ㱻��Ϊ��Բ��
			if (adata[base] > acc_threshold &&
				adata[base] > adata[base - 1] && adata[base] > adata[base + 1] &&
				adata[base] > adata[base - acols - 2] && adata[base] > adata[base + acols + 2])
				//�ѵ�ǰ��ĵ�ַѹ��Բ������centers��
				cvSeqPush(centers, &base);
		}
	}

	// ��ʾһ��accumaltor�е�ͶƱ��
	cv::Ptr<CvMat> acc_img;
	acc_img = cvCreateMat(accum->rows, accum->cols, CV_8UC1);
	cvNormalize(accum, acc_img, 0, 255, cv::NORM_MINMAX);
	cvShowImage("accum", acc_img);
	cvWaitKey(0);

	//����Բ�ĵ�����
	center_count = centers->total;
	//�������Ϊ0��˵��û�м�⵽Բ�����˳��ú���
	if (!center_count)
		return;
	//�������������Ĵ�С
	sort_buf.resize(MAX(center_count, nz_count));
	//��Բ�����з�������������
	cvCvtSeqToArray(centers, &sort_buf[0]);
	//��Բ�İ����ɴ�С��˳���������
	//����ԭ���Ǿ���icvHoughSortDescent32s��������sort_buf��Ԫ����Ϊadata�����±꣬adata�е�Ԫ�ؽ������У���adata[sort_buf[0]]��adata����Ԫ�������ģ�adata[sort_buf[center_count-1]]������Ԫ������С��
	icvHoughSortDescent32s(&sort_buf[0], center_count, adata);
	//���Բ������
	cvClearSeq(centers);
	//���ź����Բ�����·���Բ��������
	cvSeqPushMulti(centers, &sort_buf[0], center_count);
	//�����뾶�������
	dist_buf = cvCreateMat(1, nz_count, CV_32FC1);
	//�����ַָ��
	ddata = dist_buf->data.fl;

	inner_products_buf = cvCreateMat(1, nz_count, CV_32FC1);
	//�����ַָ��
	idata = inner_products_buf->data.fl;


	dr = dp;    //����Բ�뾶�ľ���ֱ���
				//���¶���Բ��֮�����С����
	min_dist = MAX(min_dist, dp);
	//��С�����ƽ��
	min_dist *= min_dist;
	// For each found possible center
	// Estimate radius and check support
	//�����ɴ�С��˳���������Բ������
	for (i = 0; i < centers->total; i++)
	{
		//��ȡ��Բ�ģ��õ��õ����ۼ��������е�ƫ����
		int ofs = *(int*)cvGetSeqElem(centers, i);
		//�õ�Բ�����ۼ����е�����λ��
		y = ofs / (acols + 2);
		x = ofs - (y)*(acols + 2);
		//Calculate circle's center in pixels
		//����Բ��������ͼ���е�����λ��
		float cx = (float)((x + 0.5f)*dp), cy = (float)((y + 0.5f)*dp);
		float start_dist, dist_sum;
		float r_best = 0;
		int max_count = 0;
		// Check distance with previously detected circles
		//�жϵ�ǰ��Բ����֮ǰȷ����Ϊ�����Բ���Ƿ�Ϊͬһ��Բ��
		for (j = 0; j < circles->total; j++)
		{
			//����������ȡ��Բ��
			float* c = (float*)cvGetSeqElem(circles, j);
			//���㵱ǰԲ������ȡ����Բ��֮��ľ��룬������߾���С���������ֵ������Ϊ����Բ����ͬһ��Բ�ģ��˳�ѭ��
			if ((c[0] - cx)*(c[0] - cx) + (c[1] - cy)*(c[1] - cy) < min_dist)
				break;
		}
		//���j < circles->total��˵����ǰ��Բ���ѱ���Ϊ��֮ǰȷ����Ϊ�����Բ����ͬһ��Բ�ģ���������Բ�ģ����������forѭ��
		if (j < circles->total)
			continue;
		// Estimate best radius
		//�ڶ��׶�
		//��ʼ��ȡԲ������nz
		cvStartReadSeq(nz, &reader);
		for (j = k = 0; j < nz_count; j++)
		{
			CvPoint pt;

			float _dx, _dy, _r2;
			CV_READ_SEQ_ELEM(pt, reader);
			_dx = cx - pt.x; _dy = cy - pt.y;
			//����2.1������Բ���ϵĵ��뵱ǰԲ�ĵľ��룬���뾶
			_r2 = _dx*_dx + _dy*_dy;

			float x_norm = _dx / pow(_r2, 0.5);
			float y_norm = _dy / pow(_r2, 0.5);

			// ���sx,dx���ǵ������ڻ�������SHIFTλ��
			const short* dxdy_row = (short*)(dxdy->data.ptr + pt.y*dxdy->step);
			short sx = dxdy_row[2 * pt.x + 0];
			short sy = dxdy_row[2 * pt.x + 1];

			//����2.2������뾶�������õ����뾶����С�뾶֮��
			if (min_radius2 <= _r2 && _r2 <= max_radius2)
			{
				//�Ѱ뾶����dist_buf��
				ddata[k] = _r2;
				sort_buf[k] = k;
				idata[k] = sx * x_norm + sy * y_norm;
				k++;
			}
		}
		//k��ʾһ���ж��ٸ�Բ���ϵĵ�
		int nz_count1 = k, start_idx = nz_count1 - 1;
		//nz_count1����0Ҳ����k����0��˵����ǰ��Բ��û������Ӧ��Բ����ζ�ŵ�ǰԲ�Ĳ���������Բ�ģ�����������Բ�ģ����������forѭ��
		if (nz_count1 == 0)
			continue;
		dist_buf->cols = nz_count1;    //�õ�Բ���ϵ�ĸ���
		cvPow(dist_buf, dist_buf, 0.5);    //��ƽ�������õ�������Բ�뾶
										   //����2.3����Բ�뾶��������
		icvHoughSortDescent32s(&sort_buf[0], nz_count1, (int*)ddata);

		dist_sum = start_dist = ddata[sort_buf[nz_count1 - 1]];
		float cur_r_dist_sum = 0.0;
		int cur_r_count = 0;
		int cur_r_grad_count = 0;
		//����2.4
		for (j = nz_count1 - 2; j >= 0; j--)
		{
			float d = ddata[sort_buf[j]];
			float inner_product = idata[sort_buf[j]];

			if (d > max_radius)
				break;
			//d��ʾ��ǰ�뾶ֵ��start_dist��ʾ��һ��ͨ������if�����º�İ뾶ֵ��dr��ʾ�뾶����ֱ��ʣ�����������뾶����֮����ھ���ֱ��ʣ�˵���������뾶һ��������ͬһ��Բ������������if�������֮�����Щ�뾶ֵ������Ϊ����ȵģ���������ͬһ��Բ
			if (d - start_dist < HOUGH_CIRCLE_RADIUS_MIN_DIST * dr)
			{
				////start_idx��ʾ��һ�ν���if���ʱ���µİ뾶������������
				//// start_idx �C j��ʾ��ǰ�õ�����ͬ�뾶���������
				////(j + start_idx)/2��ʾj��start_idx�м����
				////ȡ�м��������Ӧ�İ뾶ֵ��Ϊ��ǰ�뾶ֵr_cur��Ҳ����ȡ��Щ�뾶ֵ��ͬ��ֵ
				//float r_cur = ddata[sort_buf[(j + start_idx) / 2]];
				////�����ǰ�õ��İ뾶��ͬ�������������ֵmax_count�������if���
				//if ((start_idx - j)*r_best >= max_count*r_cur ||
				//	(r_best < FLT_EPSILON && start_idx - j >= max_count))
				//{
				//	r_best = r_cur;    //�ѵ�ǰ�뾶ֵ��Ϊ��Ѱ뾶ֵ
				//	max_count = start_idx - j;    //�������ֵ
				//}
				////���°뾶��������
				//start_dist = d;
				//start_idx = j;
				//dist_sum = 0;

				// �����޸Ĵ��룬ʹ�ü������е�Բ�Ŀ��Զ�Ӧ����뾶�����Ҿ������ٸ��㷨�Բ���������
				cur_r_count++;
				cur_r_dist_sum += d;
				// �����ݶȼ��
				if (fabs(inner_product) > HOUGH_CIRCLE_SAMEDIRECT_DEGREE * ONE) {
					cur_r_grad_count++;
				}
			}
			// ˵����ʱ�Ѿ�����һ��Բ�������Ҫ�ж���Բ�Ƿ�ϸ񣬲���Ҫ����ĳЩ�ֲ�������Ϊ��ͳ����һ��Բ
			else {
				// ����ƽ���뾶
				float r_mean = cur_r_dist_sum / cur_r_count;
				// �жϸ�Բ�Ƿ�ϸ�
				if (cur_r_count >= HOUGH_CIRCLE_INTEGRITY_DEGREE * 2 * HOUGH_MATH_PI * r_mean &&
					cur_r_grad_count >= HOUGH_CIRCLE_GRADIENT_INTEGRITY_DEGREE * cur_r_count){
					float c[3];
					c[0] = cx;    //Բ�ĵĺ�����
					c[1] = cy;    //Բ�ĵ�������
					c[2] = (float)r_mean;    //����Ӧ��Բ�İ뾶
					cvSeqPush(circles, c);    //ѹ������circles��
											  //����õ���Բ������ֵ�����˳��ú���
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
	//Ϊ��������㾫�ȣ�����һ����ֵ��λ����
	const int SHIFT = 10, ONE = 1 << SHIFT;
	//����ˮƽ�ݶȺʹ�ֱ�ݶȾ���ĵ�ַָ��
	cv::Ptr<CvMat> dx, dy;
	//�����Եͼ���ۼ�������Ͱ뾶�������ĵ�ַָ��
	cv::Ptr<CvMat> edges, accum, dist_buf;
	//������������
	std::vector<int> sort_buf;
	cv::Ptr<CvMemStorage> storage;

	int x, y, i, j, k, center_count, nz_count;
	//���ȼ������С�뾶�����뾶��ƽ��
	float min_radius2 = (float)min_radius*min_radius;
	float max_radius2 = (float)max_radius*max_radius;
	int rows, cols, arows, acols;
	int astep, *adata;
	float* ddata;
	//nz��ʾԲ�����У�centers��ʾԲ������
	CvSeq *nz, *centers;
	float idp, dr;
	CvSeqReader reader;
	//����һ����Եͼ�����
	edges = cvCreateMat(img->rows, img->cols, CV_8UC1);
	//��һ�׶�
	//����1.1����canny��Ե����㷨�õ�����ͼ��ı�Եͼ��
	cvCanny(img, edges, MAX(canny_threshold / 2, 1), canny_threshold, 3);
	cvShowImage("edges", edges);
	cvWaitKey(0);
	//��������ͼ���ˮƽ�ݶ�ͼ��ʹ�ֱ�ݶ�ͼ��
	dx = cvCreateMat(img->rows, img->cols, CV_16SC1);
	dy = cvCreateMat(img->rows, img->cols, CV_16SC1);
	//����1.2����Sobel���ӷ�����ˮƽ�ݶȺʹ�ֱ�ݶ�
	cvSobel(img, dx, 1, 0, 3);
	cvSobel(img, dy, 0, 1, 3);
	// ȷ���ۼ�������ķֱ��ʲ�С��1
	if (dp < 1.f)
		dp = 1.f;
	//�ֱ��ʵĵ���
	idp = 1.f / dp;
	//���ݷֱ��ʣ������ۼ�������
	accum = cvCreateMat(cvCeil(img->rows*idp) + 2, cvCeil(img->cols*idp) + 2, CV_32SC1);
	//��ʼ���ۼ���Ϊ0
	cvZero(accum);
	//�����������У�
	storage = cvCreateMemStorage();
	nz = cvCreateSeq(CV_32SC2, sizeof(CvSeq), sizeof(CvPoint), storage);
	centers = cvCreateSeq(CV_32SC1, sizeof(CvSeq), sizeof(int), storage);

	rows = img->rows;    //ͼ��ĸ�
	cols = img->cols;    //ͼ��Ŀ�
	arows = accum->rows - 2;    //�ۼ����ĸ�
	acols = accum->cols - 2;    //�ۼ����Ŀ�
	adata = accum->data.i;    //�ۼ����ĵ�ַָ��
	astep = accum->step / sizeof(adata[0]); // �ۼ����Ĳ���
											// Accumulate circle evidence for each edge pixel
											//����1.3���Ա�Եͼ������ۼӺ�
	for (y = 0; y < rows; y++)
	{
		//��ȡ����Եͼ��ˮƽ�ݶ�ͼ��ʹ�ֱ�ݶ�ͼ���ÿ�е��׵�ַ
		const uchar* edges_row = edges->data.ptr + y*edges->step;
		const short* dx_row = (const short*)(dx->data.ptr + y*dx->step);
		const short* dy_row = (const short*)(dy->data.ptr + y*dy->step);

		for (x = 0; x < cols; x++)
		{
			float vx, vy;
			int sx, sy, x0, y0, x1, y1, r;
			CvPoint pt;
			//��ǰ��ˮƽ�ݶ�ֵ�ʹ�ֱ�ݶ�ֵ
			vx = dx_row[x];
			vy = dy_row[x];
			//�����ǰ�����ز��Ǳ�Ե�㣬����ˮƽ�ݶ�ֵ�ʹ�ֱ�ݶ�ֵ��Ϊ0�������ѭ������Ϊ������������������õ�һ������Բ���ϵĵ�
			if (!edges_row[x] || (vx == 0 && vy == 0))
				continue;
			//���㵱ǰ����ݶ�ֵ
			float mag = sqrt(vx*vx + vy*vy);
			assert(mag >= 1);
			//����ˮƽ�ʹ�ֱ��λ����
			sx = cvRound((vx*idp)*ONE / mag);
			sy = cvRound((vy*idp)*ONE / mag);
			//�ѵ�ǰ������궨λ���ۼ�����λ����
			x0 = cvRound((x*idp)*ONE);
			y0 = cvRound((y*idp)*ONE);
			// Step from min_radius to max_radius in both directions of the gradient
			//���ݶȵ����������Ͻ���λ�ƣ������ۼ�������ͶƱ�ۼ�
			for (int k1 = 0; k1 < 2; k1++)
			{
				//��ʼһ��λ�Ƶ�����
				//λ����������С�뾶���Ӷ���֤��������Բ�İ뾶һ���Ǵ�����С�뾶
				x1 = x0 + min_radius * sx;
				y1 = y0 + min_radius * sy;
				//���ݶȵķ�����λ��
				// r <= max_radius��֤��������Բ�İ뾶һ����С�����뾶
				for (r = min_radius; r <= max_radius; x1 += sx, y1 += sy, r++)
				{
					int x2 = x1 >> SHIFT, y2 = y1 >> SHIFT;
					//���λ�ƺ�ĵ㳬�����ۼ�������ķ�Χ�����˳�
					if ((unsigned)x2 >= (unsigned)acols ||
						(unsigned)y2 >= (unsigned)arows)
						break;
					//���ۼ�������Ӧλ���ϼ�1
					adata[y2*astep + x2]++;
				}
				//��λ��������Ϊ������
				sx = -sx; sy = -sy;
			}
			//������ͼ���еĵ�ǰ�㣨��Բ���ϵĵ㣩������ѹ������Բ������nz��
			pt.x = x; pt.y = y;
			cvSeqPush(nz, &pt);
		}
	}
	//����Բ�ܵ������
	nz_count = nz->total;
	//�������Ϊ0��˵��û�м�⵽Բ�����˳��ú���
	if (!nz_count)
		return;
	//Find possible circle centers
	//����1.4��1.5�����������ۼ��������ҵ����ܵ�Բ��
	for (y = 1; y < arows - 1; y++)
	{
		for (x = 1; x < acols - 1; x++)
		{
			int base = y*(acols + 2) + x;
			//�����ǰ��ֵ������ֵ������4�������������ֵ����õ㱻��Ϊ��Բ��
			if (adata[base] > acc_threshold &&
				adata[base] > adata[base - 1] && adata[base] > adata[base + 1] &&
				adata[base] > adata[base - acols - 2] && adata[base] > adata[base + acols + 2])
				//�ѵ�ǰ��ĵ�ַѹ��Բ������centers��
				cvSeqPush(centers, &base);
		}
	}
	//����Բ�ĵ�����
	center_count = centers->total;
	//�������Ϊ0��˵��û�м�⵽Բ�����˳��ú���
	if (!center_count)
		return;
	//�������������Ĵ�С
	sort_buf.resize(MAX(center_count, nz_count));
	//��Բ�����з�������������
	cvCvtSeqToArray(centers, &sort_buf[0]);
	//��Բ�İ����ɴ�С��˳���������
	//����ԭ���Ǿ���icvHoughSortDescent32s��������sort_buf��Ԫ����Ϊadata�����±꣬adata�е�Ԫ�ؽ������У���adata[sort_buf[0]]��adata����Ԫ�������ģ�adata[sort_buf[center_count-1]]������Ԫ������С��
	icvHoughSortDescent32s(&sort_buf[0], center_count, adata);
	//���Բ������
	cvClearSeq(centers);
	//���ź����Բ�����·���Բ��������
	cvSeqPushMulti(centers, &sort_buf[0], center_count);
	//�����뾶�������
	dist_buf = cvCreateMat(1, nz_count, CV_32FC1);
	//�����ַָ��
	ddata = dist_buf->data.fl;

	dr = dp;    //����Բ�뾶�ľ���ֱ���
				//���¶���Բ��֮�����С����
	min_dist = MAX(min_dist, dp);
	//��С�����ƽ��
	min_dist *= min_dist;
	// For each found possible center
	// Estimate radius and check support
	//�����ɴ�С��˳���������Բ������
	for (i = 0; i < centers->total; i++)
	{
		//��ȡ��Բ�ģ��õ��õ����ۼ��������е�ƫ����
		int ofs = *(int*)cvGetSeqElem(centers, i);
		//�õ�Բ�����ۼ����е�����λ��
		y = ofs / (acols + 2);
		x = ofs - (y)*(acols + 2);
		//Calculate circle's center in pixels
		//����Բ��������ͼ���е�����λ��
		float cx = (float)((x + 0.5f)*dp), cy = (float)((y + 0.5f)*dp);
		float start_dist, dist_sum;
		float r_best = 0;
		int max_count = 0;
		// Check distance with previously detected circles
		//�жϵ�ǰ��Բ����֮ǰȷ����Ϊ�����Բ���Ƿ�Ϊͬһ��Բ��
		for (j = 0; j < circles->total; j++)
		{
			//����������ȡ��Բ��
			float* c = (float*)cvGetSeqElem(circles, j);
			//���㵱ǰԲ������ȡ����Բ��֮��ľ��룬������߾���С���������ֵ������Ϊ����Բ����ͬһ��Բ�ģ��˳�ѭ��
			if ((c[0] - cx)*(c[0] - cx) + (c[1] - cy)*(c[1] - cy) < min_dist)
				break;
		}
		//���j < circles->total��˵����ǰ��Բ���ѱ���Ϊ��֮ǰȷ����Ϊ�����Բ����ͬһ��Բ�ģ���������Բ�ģ����������forѭ��
		if (j < circles->total)
			continue;
		// Estimate best radius
		//�ڶ��׶�
		//��ʼ��ȡԲ������nz
		cvStartReadSeq(nz, &reader);
		for (j = k = 0; j < nz_count; j++)
		{
			CvPoint pt;
			float _dx, _dy, _r2;
			CV_READ_SEQ_ELEM(pt, reader);
			_dx = cx - pt.x; _dy = cy - pt.y;
			//����2.1������Բ���ϵĵ��뵱ǰԲ�ĵľ��룬���뾶
			_r2 = _dx*_dx + _dy*_dy;
			//����2.2������뾶�������õ����뾶����С�뾶֮��
			if (min_radius2 <= _r2 && _r2 <= max_radius2)
			{
				//�Ѱ뾶����dist_buf��
				ddata[k] = _r2;
				sort_buf[k] = k;
				k++;
			}
		}
		//k��ʾһ���ж��ٸ�Բ���ϵĵ�
		int nz_count1 = k, start_idx = nz_count1 - 1;
		//nz_count1����0Ҳ����k����0��˵����ǰ��Բ��û������Ӧ��Բ����ζ�ŵ�ǰԲ�Ĳ���������Բ�ģ�����������Բ�ģ����������forѭ��
		if (nz_count1 == 0)
			continue;
		dist_buf->cols = nz_count1;    //�õ�Բ���ϵ�ĸ���
		cvPow(dist_buf, dist_buf, 0.5);    //��ƽ�������õ�������Բ�뾶
										   //����2.3����Բ�뾶��������
		icvHoughSortDescent32s(&sort_buf[0], nz_count1, (int*)ddata);

		dist_sum = start_dist = ddata[sort_buf[nz_count1 - 1]];
		//����2.4
		for (j = nz_count1 - 2; j >= 0; j--)
		{
			float d = ddata[sort_buf[j]];

			if (d > max_radius)
				break;
			//d��ʾ��ǰ�뾶ֵ��start_dist��ʾ��һ��ͨ������if�����º�İ뾶ֵ��dr��ʾ�뾶����ֱ��ʣ�����������뾶����֮����ھ���ֱ��ʣ�˵���������뾶һ��������ͬһ��Բ������������if�������֮�����Щ�뾶ֵ������Ϊ����ȵģ���������ͬһ��Բ
			if (d - start_dist > dr)
			{
				//start_idx��ʾ��һ�ν���if���ʱ���µİ뾶������������
				// start_idx �C j��ʾ��ǰ�õ�����ͬ�뾶���������
				//(j + start_idx)/2��ʾj��start_idx�м����
				//ȡ�м��������Ӧ�İ뾶ֵ��Ϊ��ǰ�뾶ֵr_cur��Ҳ����ȡ��Щ�뾶ֵ��ͬ��ֵ
				float r_cur = ddata[sort_buf[(j + start_idx) / 2]];
				//�����ǰ�õ��İ뾶��ͬ�������������ֵmax_count�������if���
				if ((start_idx - j)*r_best >= max_count*r_cur ||
					(r_best < FLT_EPSILON && start_idx - j >= max_count))
				{
					r_best = r_cur;    //�ѵ�ǰ�뾶ֵ��Ϊ��Ѱ뾶ֵ
					max_count = start_idx - j;    //�������ֵ
				}
				//���°뾶��������
				start_dist = d;
				start_idx = j;
				dist_sum = 0;
			}
			dist_sum += d;
		}
		// Check if the circle has enough support
		//����2.5������ȷ�����
		//�����ͬ�뾶����������������ֵ
		if (max_count > acc_threshold)
		{
			float c[3];
			c[0] = cx;    //Բ�ĵĺ�����
			c[1] = cy;    //Բ�ĵ�������
			c[2] = (float)r_best;    //����Ӧ��Բ�İ뾶
			cvSeqPush(circles, c);    //ѹ������circles��
									  //����õ���Բ������ֵ�����˳��ú���
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
	int circles_max = INT_MAX;    //������Բ�ε���������Ϊ�����
	//canny��Ե�����˫��ֵ�еĸ���ֵ
	int canny_threshold = cvRound(param1);
	//�ۼ�����ֵ
	int acc_threshold = cvRound(param2);

	img = cvGetMat(img, &stub);
	//ȷ������ͼ���ǻҶ�ͼ��
	if (!CV_IS_MASK_ARR(img))
		CV_Error(CV_StsBadArg, "The source image must be 8-bit, single-channel");
	//�ڴ�ռ��Ƿ����
	if (!circle_storage)
		CV_Error(CV_StsNullPtr, "NULL destination");
	//ȷ����������ȷ��
	if (dp <= 0 || min_dist <= 0 || canny_threshold <= 0 || acc_threshold <= 0)
		CV_Error(CV_StsOutOfRange, "dp, min_dist, canny_threshold and acc_threshold must be all positive numbers");
	//Բ����С�뾶Ҫ����0
	min_radius = MAX(min_radius, HOUGH_CIRCLE_RADIUS_MIN);
	//Բ�����뾶���С�ڵ���0���������뾶Ϊͼ���ͳ��ȵ����ֵ��
	//������뾶С����С�뾶���������뾶Ϊ��С�뾶���������صĿ��
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
	//ѡ�������㷨���Բ��Ŀǰֻ��2-1����任
	switch (method)
	{
	case CV_HOUGH_GRADIENT:
		//����icvHoughCirclesGradient����
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
	//���Բ
	return result;
}


void myHoughCircles( cv::InputArray _image, cv::OutputArray _circles,
int method, double dp, double min_dist,
double param1, double param2,
int minRadius, int maxRadius )
{
	//����һ���ڴ�
	cv::Ptr<CvMemStorage> storage = cvCreateMemStorage(0);
	cv::Mat image = _image.getMat();    //��ȡ����ͼ�����
	CvMat c_image = image;    //����ת��
							  //����cvHoughCircles����
	CvSeq* seq = cvHoughCircles(&c_image, storage, method,
		dp, min_dist, param1, param2, minRadius, maxRadius);
	//������ת��Ϊ����
	seqToMat(seq, _circles);
}
