#include <stdlib.h>
#include <stdio.h>

float train[][2] = {
	{0, 0},
	{1, 2},
	{2, 4},
	{3, 6},
	{4, 8},
	{5, 10}
};

#define train_size (int)(sizeof(train) / sizeof(train[0]))

float rand_float()
{
	return (float) rand() / (float) RAND_MAX;
}

float mse(float w, float b)
{
	float error = 0;

	for (int i = 0; i < train_size; i++)
	{
		float y = w * train[i][0] + b;
		float d = y - train[i][1];
		error += d*d;
	}

	return error / train_size;
}

int main()
{
	srand(101);
	int epocs = 30000;

	float w = rand_float() * 10;
	float b = rand_float() * 10;
	float c;

	for (int _ = 0; _ < epocs; _++)
	{
		float eps = 1e-3;
		float lr = 1e-3;

		float dw = (mse(w + eps, b) - mse(w, b)) / eps;
		float db = (mse(w, b + eps) - mse(w, b)) / eps;

		w -= dw * lr;
		b -= db * lr;

		c = mse(w, b);
		printf("%f\n", c);
	}

	printf("w = %f, b = %f, c = %f\n", w, b, c);

}
