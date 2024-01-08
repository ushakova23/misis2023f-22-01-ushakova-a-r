#ifndef header_h
#define header_h


#include <opencv2/opencv.hpp>

void centralProjection(cv::Mat& inputImage) {
    // Проверка на правильность размеров входного изображения
    if (inputImage.rows != inputImage.cols || inputImage.channels() != 3) {
        std::cout << "Некорректные размеры входного изображения. Ожидается квадратное цветное изображение." << std::endl;
        return;
    }

    // Создание выходного изображения
    cv::Mat outputImage(inputImage.rows, inputImage.cols, CV_8UC3, cv::Scalar(0, 0, 0));

    // Расчет главной диагонали
    int diagonalLength = inputImage.rows * std::sqrt(3);

    // Проход по каждой точке входного изображения
    for (int x = 0; x < inputImage.cols; x++) {
        for (int y = 0; y < inputImage.rows; y++) {
            // Вычисление координаты точки в 3D пространстве
            double x3D = x - inputImage.cols / 2;
            double y3D = y - inputImage.rows / 2;
            double z3D = diagonalLength / 2;

            // Проекция точки на плоскость, перпендикулярную главной диагонали
            double scaleFactor = z3D / (z3D - y3D);
            double projectedX = x3D * scaleFactor;
            double projectedY = x3D * scaleFactor;

            // Преобразование координат в индексы изображения
            int outputX = static_cast<int>(projectedX + outputImage.cols / 2);
            int outputY = static_cast<int>(projectedY + outputImage.rows / 2);

            // Установка значения пикселя в выходном изображении
            outputImage.at<cv::Vec3b>(outputY, outputX) = inputImage.at<cv::Vec3b>(y, x);
        }
    }

    // Отображение выходного изображения
    cv::imshow("Центральная проекция", outputImage);
    cv::waitKey(0);
}

/*int main() {
    // Загрузка изображения
    cv::Mat inputImage = cv::imread("input_image.jpg", cv::IMREAD_COLOR);

    // Проверка на успешную загрузку изображения
    if (inputImage.empty()) {
        std::cout << "Не удалось загрузить изображение." << std::endl;
        return -1;
    }

    // Вызов функции центральной проекции
    centralProjection(inputImage);

    return 0;
}*/
#endif header_h
