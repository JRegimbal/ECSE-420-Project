#pragma once

enum CellType {
	BLANK=0,
	NEURON,
	AXON,
	DENDRITE,
	CT_INVALID
};

enum Direction {
	NONE=0,
	NORTH,
	EAST,
	SOUTH,
	WEST,
	UP,
	DOWN,
	D_INVALID
};

enum Instruction {
	STOP=0,
	GROW_STRAIGHT,
	TURN_LEFT,
	TURN_RIGHT,
	SPLIT_LEFT,
	SPLIT_RIGHT,
	I_INVALID
};

class Cell {
protected:
	CellType type;
	Instruction chromosome;
	Direction gate;
	bool value = false;	// Value for signaling phase
	bool grown = false;	// If this cell has followed the growth instruction. This must only happen once.

public:
	Cell() : type(CellType::BLANK), chromosome(Instruction::I_INVALID), gate(Direction::D_INVALID) {}
	Cell(Instruction inst) : type(CellType::BLANK), chromosome(inst), gate(Direction::D_INVALID) {}
	Cell(CellType type, Instruction inst, Direction gate) : type(type), chromosome(inst), gate(gate) {}

	static Direction oppositeDirection(Direction dir) {
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

	Instruction getInstruction() { return this->chromosome; }

	void growCell(CellType type, Direction growthDirection) {
		if (this->type != CellType::BLANK) return; // Only blank cells can grow into new cells!
		// Initialize based on CellType
		switch (type) {
		case CellType::AXON:
			// Gate is in direction of growth
			this->type = type;
			this->gate = growthDirection;
			break;
		case CellType::DENDRITE:
			// Gate points toward the signaling cell (i.e. opposite of growth direction)
			this->type = type;
			this->gate = Cell::oppositeDirection(growthDirection);
			break;
		case CellType::NEURON:
			// THIS SHOULD NOT HAPPEN. NEURONS USE A SPECIFIC FUNCTION TO GROW.
		default:
			this->type = CellType::CT_INVALID;
			this->gate = Direction::D_INVALID;
			break;
		}
	}
};
