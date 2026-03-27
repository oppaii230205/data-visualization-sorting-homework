#include <algorithm>
#include <direct.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

enum class InputOrder
{
	RandomizedData,
	SortedData,
	ReverseSortedData,
	NearlySortedData,
	DuplicatedData
};

template <class T>
void Swap(T &a, T &b)
{
	T x = a;
	a = b;
	b = x;
}

void GenerateRandomData(std::vector<int> &a, std::mt19937 &rng)
{
	std::uniform_int_distribution<int> dist(0, static_cast<int>(a.size()) - 1);
	for (size_t i = 0; i < a.size(); i++)
	{
		a[i] = dist(rng);
	}
}

void GenerateSortedData(std::vector<int> &a)
{
	for (size_t i = 0; i < a.size(); i++)
	{
		a[i] = static_cast<int>(i);
	}
}

void GenerateReverseData(std::vector<int> &a)
{
	for (size_t i = 0; i < a.size(); i++)
	{
		a[i] = static_cast<int>(a.size() - 1 - i);
	}
}

void GenerateNearlySortedData(std::vector<int> &a, std::mt19937 &rng)
{
	GenerateSortedData(a);
	const int n = static_cast<int>(a.size());
	const int numberOfSwaps = std::max(10, n / 100);
	std::uniform_int_distribution<int> indexDist(0, n - 1);

	for (int i = 0; i < numberOfSwaps; i++)
	{
		int r1 = indexDist(rng);
		int r2 = indexDist(rng);
		Swap(a[r1], a[r2]);
	}
}

// Duplicate-heavy data: many repeated values by restricting numbers to a very small range.
void GenerateDuplicatedData(std::vector<int> &a, std::mt19937 &rng)
{
	const int n = static_cast<int>(a.size());
	const int distinctValues = std::max(2, n / 50);
	std::uniform_int_distribution<int> dist(0, distinctValues - 1);

	for (size_t i = 0; i < a.size(); i++)
	{
		a[i] = dist(rng);
	}
}

void GenerateData(std::vector<int> &a, InputOrder inputOrder, std::mt19937 &rng)
{
	switch (inputOrder)
	{
	case InputOrder::RandomizedData:
		GenerateRandomData(a, rng);
		break;
	case InputOrder::SortedData:
		GenerateSortedData(a);
		break;
	case InputOrder::ReverseSortedData:
		GenerateReverseData(a);
		break;
	case InputOrder::NearlySortedData:
		GenerateNearlySortedData(a, rng);
		break;
	case InputOrder::DuplicatedData:
		GenerateDuplicatedData(a, rng);
		break;
	default:
		std::cerr << "Error: unknown data type!\n";
	}
}

std::string InputOrderToString(InputOrder inputOrder)
{
	switch (inputOrder)
	{
	case InputOrder::RandomizedData:
		return "random";
	case InputOrder::SortedData:
		return "sorted";
	case InputOrder::ReverseSortedData:
		return "reversed";
	case InputOrder::NearlySortedData:
		return "nearly_sorted";
	case InputOrder::DuplicatedData:
		return "many_duplicates";
	default:
		return "unknown";
	}
}

bool WriteArrayToFile(const std::string &filePath, const std::vector<int> &a)
{
	std::ofstream out(filePath);
	if (!out.is_open())
	{
		return false;
	}

	out << a.size() << '\n';
	for (size_t i = 0; i < a.size(); i++)
	{
		out << a[i];
		if (i + 1 < a.size())
		{
			out << ' ';
		}
	}
	out << '\n';
	return true;
}

std::string JoinPath(const std::string &dir, const std::string &file)
{
	if (dir.empty())
	{
		return file;
	}
	char last = dir.back();
	if (last == '/' || last == '\\')
	{
		return dir + file;
	}
	return dir + "\\" + file;
}

int main(int argc, char *argv[])
{
	std::string outputDir = "input_data";
	int filesPerTypePerSize = 3;

	if (argc >= 2)
	{
		outputDir = argv[1];
	}
	if (argc >= 3)
	{
		filesPerTypePerSize = std::stoi(argv[2]);
	}

	if (_mkdir(outputDir.c_str()) != 0)
	{
		std::ofstream probe(JoinPath(outputDir, "__probe__.tmp"));
		if (!probe.is_open())
		{
			std::cerr << "Cannot create or access output folder: " << outputDir << '\n';
			return 1;
		}
		probe.close();
		remove(JoinPath(outputDir, "__probe__.tmp").c_str());
	}

	std::vector<int> sizes = {100, 1000, 10000, 100000};
	std::vector<InputOrder> allOrders = {
		InputOrder::RandomizedData,
		InputOrder::NearlySortedData,
		InputOrder::DuplicatedData,
		InputOrder::ReverseSortedData};

	std::random_device rd;
	std::mt19937 rng(rd());

	int generatedCount = 0;
	for (int n : sizes)
	{
		for (InputOrder order : allOrders)
		{
			for (int fileIndex = 1; fileIndex <= filesPerTypePerSize; fileIndex++)
			{
				std::vector<int> a(n);
				GenerateData(a, order, rng);

				std::ostringstream fileName;
				fileName << "n" << n << "_" << InputOrderToString(order) << "_"
						 << std::setw(2) << std::setfill('0') << fileIndex << ".txt";

				std::string filePath = JoinPath(outputDir, fileName.str());
				if (!WriteArrayToFile(filePath, a))
				{
					std::cerr << "Cannot write file: " << filePath << '\n';
					return 1;
				}

				generatedCount++;
			}
		}
	}

	std::cout << "Generated " << generatedCount << " files in folder: " << outputDir << '\n';
	std::cout << "Usage: DataGenerator.exe [output_folder] [files_per_type_per_size]\n";
	return 0;
}