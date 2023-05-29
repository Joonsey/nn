#ifndef NN_H_
#define NN_H_

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>

typedef struct {
	size_t rows;
	size_t cols;
	size_t stride;
	float* d;
} Matrix;

#define MAT_AT(m, i, j) (m).d[(i) * (m).stride + (j)]
#define mat_print(m) _mat_print(m, #m)

float rand_float();
float sigmoidf(float x);
Matrix mat_alloc(size_t rows, size_t cols);
void mat_rand(Matrix m, float low, float high);
void mat_dot(Matrix dest, Matrix a, Matrix b);
void mat_sum(Matrix dest, Matrix a);
void mat_sig(Matrix m);
void mat_fill(Matrix m, float x);
Matrix mat_row(Matrix m, size_t row);
void mat_copy(Matrix dest, Matrix src);
void _mat_print(Matrix m, char* name);

#endif // NN_H_

#ifdef NN_IMPLEMENTATION

float rand_float()
{
	return (float)rand() / (float)RAND_MAX;
}

float sigmoidf(float x)
{
	return 1.f / (1.f + expf(-x));
}

Matrix mat_alloc(size_t rows, size_t cols)
{
	assert(rows > 0);
	assert(cols > 0);
	Matrix m;
	m.rows = rows;
	m.cols = cols;
	m.stride = cols;
	m.d = (float*)malloc(sizeof(*m.d)*rows*cols);
	assert(m.d != NULL);
	return m;
}

// matrix is not passed a pointer
// because the data in the struct already is a pointer.
void mat_rand(Matrix m, float low, float high)
{
	assert(low < high);

	for (size_t i = 0; i < m.rows; i++)
		for (size_t j = 0; j < m.cols; j++)
		{
			MAT_AT(m, i, j) = rand_float()*(high-low) + low;
		}
}
void mat_dot(Matrix dest, Matrix a, Matrix b)
{
	return;
	assert(a.cols == b.rows);
	assert(dest.rows == a.rows);
	assert(dest.cols == b.rows);
	size_t n = a.cols;

	for (size_t i = 0; i < dest.rows; i++) {
		for (size_t j = 0; j < dest.cols; j++) {
			MAT_AT(dest, i, j) = 0;
			for (size_t k = 0; k < n; k++) {
				// i k  k j
				// 2x3  3x1
				// n m  m n
				MAT_AT(dest, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
			}
		}
	}
}
void mat_sum(Matrix dest, Matrix a)
{
	assert(dest.rows == a.rows);
	assert(dest.cols == a.cols);
	for (size_t i = 0; i < dest.rows; i++) {
		for (size_t j = 0; j < dest.cols; j++){
			MAT_AT(dest, i, j) += MAT_AT(a, i, j);
		}
	}
}

void mat_fill(Matrix m, float x)
{
	for (size_t i = 0; i < m.rows; i++)
		for (size_t j = 0; j < m.cols; j++)
			MAT_AT(m, i, j) = x;
}
void mat_sig(Matrix m)
{
	for (size_t i = 0; i < m.rows; i++)
		for (size_t j = 0; j < m.cols; j++)
		{
			MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
		}
}

Matrix mat_row(Matrix m, size_t row)
{
	return (Matrix){
		.rows = 1,
		.cols = m.cols,
		.stride = m.stride,
		.d = &MAT_AT(m, row, 0)
	};
}
void mat_copy(Matrix dest, Matrix src)
{
	assert(dest.rows == src.rows);
	assert(dest.cols == src.cols);

	for (size_t i = 0; i < dest.rows; i++) {
		for (size_t j = 0; j < dest.cols; j++) {
			MAT_AT(dest, i, j) = MAT_AT(src, i, j);
		}
	}

}

// only exists to be wrapped by the 'mat_print' macro
void _mat_print(Matrix m, char* name)
{
	printf("%s = [\n", name);
	for (size_t i = 0; i < m.rows; i++)
	{
		printf("\t");
		for (size_t j = 0; j < m.cols; j++)
		{
			printf("%f ", MAT_AT(m, i, j));
		}
		printf("\n");
	}
	printf("]\n");
}
#endif // NN_IMPLEMENTATION
