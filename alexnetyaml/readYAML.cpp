#include <yaml-cpp/yaml.h>
#include <vector>

YAML::Node alexnet = YAML::LoadFile ("../alexnet_noweight.yaml");

int main() {
	int filter_size = alexnet["conv1"]["filter_size"].as<int>();
	// std::vector<int> weights = alexnet["conv1"]["weights"].as<std::vector<int> >();
}