# Target Detection on RLI

## Описание

## Как пользоваться

## Запуск

1. Скачивание репозитория

2. Проверка наличия установленного `uv`

    ***Для `MacOS`***

    ```zsh
    brew install uv
    ```

    или

    ```zsh
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

    ***Для `Debian`/`Ubuntu`***

    ```zsh
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

    или

    ```sh
    wget -qO- https://astral.sh/uv/install.sh | sh
    ```

    ***Для `Windows`***

    ```powershell
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

    или

    ```powershell
    winget install astral.uv
    ```

3. Запуск приложения

```sh
uv run --sync python main.py
```
