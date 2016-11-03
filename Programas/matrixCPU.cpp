#include <iostream>
#include <omp.h>
#include <string>
#include <string.h>
#include <fstream>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace std;

float MultMatixSeq(float * m1,float * m2,float * m3, unsigned long int N){
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			for (int k = 0; k < N; k++)
			{				
				m3[i*N+j] += m1[i*N+k]*m2[k*N+j];
			}
		}
	}
}

float MultMatixSeq(double * m1,double * m2,double * m3, unsigned long int N){
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			for (int k = 0; k < N; k++)
			{				
				m3[i*N+j] += m1[i*N+k]*m2[k*N+j];
			}
		}
	}
}

float MultMatixPara(float * m1,float * m2,float * m3, unsigned long int N, const int num, const int BLOCK_SIZE){
	for (int i = 0; i < num; i++)
	{
		for (int j = 0; j < num; j++)
		{
			
			for (int k = 0; k < BLOCK_SIZE; k++)
			{
				#pragma omp parallel for
				for (int m = 0; m < BLOCK_SIZE; m++)
				{
					float sum = 0.0;
					for (int r = 0; r < num; r++)
					{
						for (int p = 0; p < BLOCK_SIZE; p++)
						{
							sum += m2[i*BLOCK_SIZE*N + r*BLOCK_SIZE+k*N+p]*m1[j*BLOCK_SIZE*N+r*BLOCK_SIZE+m*N+p];
						}
					}
					m3[i*BLOCK_SIZE*N+j*BLOCK_SIZE+k*N+m] = sum;
				}
			}
		}		
	}
}

double MultMatixPara(double * m1,double * m2,double * m3, unsigned long int N, const int num, const int BLOCK_SIZE){
	for (int i = 0; i < num; i++)
	{
		for (int j = 0; j < num; j++)
		{
			
			for (int k = 0; k < BLOCK_SIZE; k++)
			{
				#pragma omp parallel for
				for (int m = 0; m < BLOCK_SIZE; m++)
				{
					double sum = 0.0;
					for (int r = 0; r < num; r++)
					{
						for (int p = 0; p < BLOCK_SIZE; p++)
						{
							sum += m2[i*BLOCK_SIZE*N + r*BLOCK_SIZE+k*N+p]*m1[j*BLOCK_SIZE*N+r*BLOCK_SIZE+m*N+p];
						}
					}
					m3[i*BLOCK_SIZE*N+j*BLOCK_SIZE+k*N+m] = sum;
				}
			}
		}		
	}
}

void writeMatrix(string fileMatrix, float * matrix, unsigned long int &x) {

	string temp = fileMatrix + ".csv";
	ofstream saida(temp.c_str());
	saida << "%%MatrixMarket matrix coordinate real general" << endl;
	int cont = 0;
	for (size_t i = 0; i < x; i++)
	{
		for (size_t j = 0; j < x; j++)
		{
			if (matrix[i*x+j] != 0) {
				cont++;
			}
		}
	}
	saida << x << " " << x << " "<<cont<< endl;
	for (size_t i = 0; i < x; i++)
	{
		for (size_t j = 0; j < x; j++)
		{
			if (matrix[i*x+j] != 0) {
				saida << i + 1 << " " << j + 1 << " " << matrix[i*x+j];
				saida << endl;
			}
		}
	}
	saida.close();

}

void writeMatrix(string fileMatrix, double * matrix, unsigned long int &x) {

	string temp = fileMatrix + ".csv";
	ofstream saida(temp.c_str());
	saida << "%%MatrixMarket matrix coordinate real general" << endl;
	int cont = 0;
	for (size_t i = 0; i < x; i++)
	{
		for (size_t j = 0; j < x; j++)
		{
			if (matrix[i*x+j] != 0) {
				cont++;
			}
		}
	}
	saida << x << " " << x << " "<<cont<< endl;
	for (size_t i = 0; i < x; i++)
	{
		for (size_t j = 0; j < x; j++)
		{
			if (matrix[i*x+j] != 0) {
				saida << i + 1 << " " << j + 1 << " " << matrix[i*x+j];
				saida << endl;
			}
		}
	}
	saida.close();

}

void saveLog(double tTime,string log){
	string temp = "log.csv";
	ofstream saida(temp.c_str(),ofstream::app);
	log = to_string(tTime)+";"+log;
	saida << log << endl;
	saida.close();
}

int main(int argc, char const *argv[])
{
	
	double tTime;

	bool bSeq = false;
	bool bDouble = false;
	bool bSave = false;

	int BLOCK_SIZE = 0;
	unsigned long int  N = 0;
	float incial,fim,auxI, auxF;

	string saida,log;

	if (argc > 1) {
		for (int i = 0; i < argc; i++) {
			if (strcmp(argv[i], "-n") == 0) {
				N = atoi(argv[++i]);
			}else if (strcmp(argv[i], "-t") == 0) {
				BLOCK_SIZE = atoi(argv[++i]);
			}
			else if (strcmp(argv[i], "-seq") == 0) {
				bSeq = true;
			}
			else if (strcmp(argv[i], "-save") == 0) {
				bSave = true;
			}
			else if (strcmp(argv[i], "-d") == 0) {
				bDouble = true;
			}			
		}		
	}
	else {
		cout << "Without parameters!!!!" << endl;
		cout << "Use:\n\
		-t ${NUM} \tset size of tiles\n\
		-n ${NUM} \tset size of matrix\n\
		-seq \t run program in sequential mode\n\
		-save \t save the resulting matrix \n\
		-d \t set matrix to double precision \n\
		\n\n\
		Example: ./matrixCPU  -n 4096 -t 64\n\n\
		By defaul, this program running in parallel and with single precision!!!!"<< endl;

		return 1;
	}

	if(bDouble){
		double * m1;
		double * m2;
		double * m3;
		m1 = (double*)malloc(N*N*sizeof(double));
		if(m1 == NULL){
			cout << "Erro ao alocar M1!!" << endl;
			return 1;
		}
		m2 = (double*)malloc(N*N*sizeof(double));
		if(m2 == NULL){
			cout << "Erro ao alocar M2!!" << endl;
			return 1;
		}
		m3 = (double*)malloc(N*N*sizeof(double));
		if(m3 == NULL){
			cout << "Erro ao alocar M3!!" << endl;
			return 1;
		}
		for (int i = 0; i < N*N; ++i)
		{
			m1[i] = 1.f;
			m2[i] = 2.f;
			m3[i] = 0;
		}

		int num = N/BLOCK_SIZE;
		cout << "Processando matriz com double precision!!" << endl;

		incial = clock();
		if(bSeq){
			MultMatixSeq(m1,m2,m3,N);
			fim = clock();
			tTime = (fim-incial)/(float)CLOCKS_PER_SEC;
			cout << tTime << endl;
			FILE *f = fopen("log.csv", "a");
    		fprintf(f, "%lu;%d;%f;Double;CPU;sequential\n",N,BLOCK_SIZE,tTime );
    		fclose(f);
			free(m1);
			free(m2);
			free(m3);
		}else{
			MultMatixPara(m1,m2,m3,N,num,BLOCK_SIZE);
			fim = clock();
			tTime = ((fim-incial)/(float)CLOCKS_PER_SEC)/8;	
			FILE *f = fopen("log.csv", "a");
    		fprintf(f, "%lu;%d;%f;Double;CPU;parallel\n",N,BLOCK_SIZE,tTime );
    		fclose(f);
			free(m1);
			free(m2);
			free(m3);
		}
		

		if(bSave){		
			writeMatrix(saida,m3,N);		
		}


	}else{
		float * m1;
		float * m2;
		float * m3;

		m1 = (float*)malloc(N*N*sizeof(float));
		if(m1 == NULL){
			cout << "Erro ao alocar M1!!" << endl;
			return 1;
		}
		m2 = (float*)malloc(N*N*sizeof(float));
		if(m2 == NULL){
			cout << "Erro ao alocar M2!!" << endl;
			return 1;
		}
		m3 = (float*)malloc(N*N*sizeof(float));
		if(m3 == NULL){
			cout << "Erro ao alocar M3!!" << endl;
			return 1;
		}
		for (int i = 0; i < N*N; ++i)
		{
			m1[i] = 1.f;
			m2[i] = 2.f;
			m3[i] = 0;
		}

		int num = N/BLOCK_SIZE;
		cout << "Processando matriz com single precision!!" << endl;

		incial = clock();
		if(bSeq){
			MultMatixSeq(m1,m2,m3,N);
			fim = clock();
			tTime = (fim-incial)/(float)CLOCKS_PER_SEC;
			cout << tTime << endl;
			FILE *f = fopen("log.csv", "a");
    		fprintf(f, "%lu;%d;%f;Float;CPU;sequential\n",N,BLOCK_SIZE,tTime );
    		fclose(f);
			free(m1);
			free(m2);
			free(m3);
		}else{
			MultMatixPara(m1,m2,m3,N,num,BLOCK_SIZE);
			fim = clock();
			tTime = ((fim-incial)/(float)CLOCKS_PER_SEC)/8;	
			FILE *f = fopen("log.csv", "a");
    		fprintf(f, "%lu;%d;%f;Float;CPU;parallel\n",N,BLOCK_SIZE,tTime );
    		fclose(f);
			cout << tTime << endl;
			
			free(m1);
			free(m2);
			free(m3);
		}


		if(bSave){		
			writeMatrix(saida,m3,N);		
		}




	}

	
	
	return 0;
}
