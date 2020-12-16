#include "UserInterface.h"

int main(int argc, const char **argv)
{
    // check command-line usage
    if (argc != 2)
    {
        printf(
            "\n"
            "Usage: ./DGtest [model_url]\n"
            "\n"
            "   <model_url>: NNEF Model URL\n"
            "\n");
        return -1;
    }

    const char *model_url = argv[1];
    UserInterface UI(model_url);
    UI.startUI();

    return 0;
}