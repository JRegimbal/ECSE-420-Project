#pragma once
#include <stdio.h>

enum CellType {
	BLANK=0,
	NEURON,
	AXON,
	DENDRITE,
	CT_INVALID
};

enum Direction {
	NONE=0,
	NORTH = 1,
	EAST = 2,
	SOUTH = 4,
	WEST = 8,
	UP = 16,
	DOWN = 32,
	D_INVALID
};

enum Instruction {
	STOP=0,
	GROW_STRAIGHT = 1,
	TURN_LEFT = 2,
	TURN_RIGHT = 4,
	SPLIT_LEFT = 8,
	SPLIT_RIGHT = 16,
	I_INVALID
};

class Cell {
protected:
	//CellType type;
	Instruction chromosome;
	Direction gate;
	bool value = false;	// Value for signaling phase
	bool grown = false;	// If this cell has followed the growth instruction. This must only happen once.

public:
	CellType type;
	Cell() : type(CellType::BLANK), chromosome(Instruction::I_INVALID), gate(Direction::D_INVALID) {}
	Cell(Instruction inst) : type(CellType::BLANK), chromosome(inst), gate(Direction::D_INVALID) {}
	Cell(CellType type, Instruction inst, Direction gate) : type(type), chromosome(inst), gate(gate) {}

	__device__ static Direction rotate(Direction dir, bool cw) {
		switch (dir) {
		case Direction::NORTH:
			return cw ? Direction::EAST : Direction::WEST;
		case Direction::EAST:
			return cw ? Direction::SOUTH : Direction::NORTH;
		case Direction::SOUTH:
			return cw ? Direction::WEST : Direction::EAST;
		case Direction::WEST:
			return cw ? Direction::NORTH : Direction::SOUTH;
		default:
			return Direction::D_INVALID;
		}
	}
	__device__ static Direction oppositeDirection(Direction dir) {
		switch (dir) {
		case Direction::NORTH:
			return Direction::SOUTH;
		case Direction::SOUTH:
			return Direction::NORTH;
		case Direction::EAST:
			return Direction::WEST;
		case Direction::WEST:
			return Direction::EAST;
		case Direction::UP:
			return Direction::DOWN;
		case Direction::DOWN:
			return Direction::UP;
		default:
			return Direction::D_INVALID;
		}
	}

	__device__ CellType getType() { return this->type; }
	__device__ Direction getGate() { return this->gate; }
	__device__ bool hasGrown() { return this->grown; }
	__device__ void setGrown() { this->grown = true; }

	__device__ Instruction getInstruction() { return this->chromosome; }

	__device__ void growCell(CellType type, Direction growthDirection) {
		if (this->type != CellType::BLANK) {
			//printf("Tried to grow onto not blank cell\n");
			return; // Only blank cells can grow into new cells!
		}
		// Initialize based on CellType
		switch (type) {
		case CellType::AXON:
			// Gate is in direction of growth
			//printf("Growing Axon\n");
			this->type = type;
			this->gate = growthDirection;
			break;
		case CellType::DENDRITE:
			// Gate points toward the signaling cell (i.e. opposite of growth direction)
			//printf("Growing dendrite\n");
			this->type = type;
			this->gate = Cell::oppositeDirection(growthDirection);
			break;
		case CellType::NEURON:
			//printf("ERROR: Neuron\n");
			// THIS SHOULD NOT HAPPEN. NEURONS USE A SPECIFIC FUNCTION TO GROW.
		default:
			this->type = CellType::CT_INVALID;
			this->gate = Direction::D_INVALID;
			break;
		}
	}
};
