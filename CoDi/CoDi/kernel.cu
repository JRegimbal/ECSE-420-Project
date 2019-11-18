
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "codi_cell.h"

#include <iostream>
#include <fstream>
#include <random>

using namespace std;

void generateRandomNumbers(size_t num, unsigned int* values) {
	random_device rd;
	cout << "Entropy: " << rd.entropy() << endl;

	for (size_t i = 0; i < num; i++) {
		values[i] = rd();
	}
}

void writeChromosome(unsigned int block, unsigned int size, unsigned int * values) {
	fstream fs;
	fs.open("chromosome.h", fstream::out | fstream::trunc);
	fs << "unsigned char chromosome[] = {";
	for (unsigned int i = 0; i < size; i++) {
		if (i % block == 0) {
			fs << endl;
		}
		fs << values[i] % Instruction::I_INVALID << ", ";
	}
	fs << endl << "};";
	fs.close();
}

int main(int argc, char** argv) {
	size_t x, y, z;
	if (argc == 2) {
		x = y = z = atoi(argv[1]);
	}
	else if (argc == 4) {
		x = atoi(argv[1]);
		y = atoi(argv[2]);
		z = atoi(argv[3]);
	}
	else {
		cerr << "Unsupported number of arguments (" << argc - 1 << ")" << endl;
		return 1;
	}
	unsigned int* values = new unsigned int[x * y * z];
	generateRandomNumbers(x * y * z, values);
	writeChromosome(x, x * y * z, values);

	return 0;
}
