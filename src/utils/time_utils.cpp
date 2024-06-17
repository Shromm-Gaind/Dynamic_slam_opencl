#include "time_utils.hpp"

std::string date_time_string(     ){

    auto now = std::chrono::system_clock::now();
	auto in_time_t = std::chrono::system_clock::to_time_t(now);

	std::stringstream datetime;
	datetime << "_" << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%X_%a") << "_ ";

    std::string   datetime_str = datetime.str();
    return datetime_str;
}
