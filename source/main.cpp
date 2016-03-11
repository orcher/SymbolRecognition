#include "neuralnetwork.hpp"

#include <QApplication>
#include <QQmlApplicationEngine>
#include <qqmlcontext.h>
#include <QQuickView>
#include <QFile>
#include <QTextStream>

#include <vector>
#include <ctime>
#include <iostream>

int main(int argc, char *argv[])
{
	srand((unsigned int)time(NULL));

	std::vector<std::vector<float> > trainingSetsInputs = {
		{ 1, 3 },
		{ 2, 4 },
		{ 4, 5 },
		{ 2, 1 },
		{ 4, 1 },
		{ 5, 3 }
	};
	std::vector<std::vector<float> > trainingSetsOutputs = {
		{ 1 },
		{ 1 },
		{ 1 },
		{ 0 },
		{ 0 },
		{ 0 }
	};

	std::vector<float> realCase = { 4, 2 };

	std::vector<unsigned int> layersSizes = { 2, 10, 1 };

	QApplication a(argc, argv);

	QQmlApplicationEngine engine;

	nn::NeuralNetwork net(layersSizes);

	engine.load(QUrl("qrc:/QML/main.qml"));
	engine.rootContext()->setContextProperty("net", &net);
	//net.recognize(realCase);

	net.learn(trainingSetsInputs, trainingSetsOutputs);

	return a.exec();
}

#ifdef WIN32
#include <Windows.h>

int __stdcall WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
	return main(__argc, __argv);
}
#endif