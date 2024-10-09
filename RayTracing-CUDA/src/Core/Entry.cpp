#pragma once
#include "App.h"

int main()
{
	auto app = new RayTracing::App();
	app->Run();
	delete app;
}