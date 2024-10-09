#pragma once
#include <iostream>

// ANSIÑÕÉ«´úÂë
#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"

#define RAY_ERROR(...)		std::cout << RED	<< "RayTracing: " << (__VA_ARGS__) << RESET << std::endl
#define RAY_WARNING(...)	std::cout << YELLOW << "RayTracing: " << (__VA_ARGS__) << RESET << std::endl
#define RAY_INFO(...)		std::cout << GREEN	<< "RayTracing: " << (__VA_ARGS__) << RESET << std::endl