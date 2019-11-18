
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "codi_cell.h"
#include "chromosome.h"

#include <iostream>
#include <fstream>
#include <random>

using namespace std;

const float neuronSeedP = 0.10;
const unsigned int sig = 100;

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
	
	Cell* cells = (Cell *)malloc(x * y * z * sizeof(Cell));
	Cell* device_cells = NULL;
	random_device rd;
	// Initialize cells.
	for (unsigned int i = 0; i < x * y * z; i++) {
		// Will this be a neuron?
		bool neuron = (rd() % sig) < (unsigned int)(neuronSeedP * sig);
		// These lines initialize Cell objects continuouslyl in memory (so we can actually memcpy them)
		if ((rd() % sig) < (unsigned int)(neuronSeedP * sig)) {
			new (&(cells[i])) Cell(CellType::NEURON, (Instruction) chromosome[i], Direction::NORTH);
		}
		else {
			new (&(cells[i])) Cell((Instruction) chromosome[i]);
		}
	}

	// Allocate cuda cell space
	cudaMalloc((void**)& device_cells, x * y * z * sizeof(Cell));
	// Copy initialized grid to device
	cudaMemcpy(device_cells, cells, x * y * z * sizeof(Cell), cudaMemcpyHostToDevice);

	// Growth Stuff

	cudaFree(device_cells);
	free(cells);

	return 0;
}
