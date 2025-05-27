#include <iostream>
#include "vector_add.hpp"
#include "brightness.hpp"

void showMenu() {
    std::cout << "Выберите задачу:\n"
              << "1 - Сложение векторов (CPU + GPU)\n"
              << "2 - Увеличение яркости изображения (CPU + GPU)\n"
              << "Введите номер задачи: ";
}

int main() {
    int choice = 0;
    showMenu();

    if (!(std::cin >> choice)) {
        std::cerr << "Ошибка ввода: ожидалось число\n";
        return 1;
    }

    switch (choice) {
        case 1:
            runVectorAddition();
            break;
        case 2:
            runBrightnessIncrease();
            break;
        default:
            std::cout << "Неверный выбор. Завершение.\n";
            break;
    }

    return 0;
}
