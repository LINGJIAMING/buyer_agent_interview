from app import create_app


def main():
    print("🚀 正在启动买手 Agent...")
    app = create_app()
    print("--- 💼 买手总监已就绪（输入 'exit' 退出）---")

    while True:
        try:
            user_input = input("商家: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("已退出。")
            break

        response = app.chat(user_input)
        print(f"买手: {response}")


if __name__ == "__main__":
    main()