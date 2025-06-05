import serial
import time
import threading


def continuous_read(ser):
    """在单独线程中持续读取串口数据"""
    while True:
        try:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                if line:
                    print(f"Get Code from Arduino: {line}")
        except Exception as e:
            print(f"error: {e}")
            break
        time.sleep(0.01)


try:
    # 连接Arduino
    arduino = serial.Serial('COM8', 9600, timeout=1)
    print("Arduino succesfully connected!")

    # 等待Arduino重置
    time.sleep(3)

    # 清空缓冲区
    arduino.reset_input_buffer()
    arduino.reset_output_buffer()

    # 启动读取线程
    read_thread = threading.Thread(target=continuous_read, args=(arduino,))
    read_thread.daemon = True  # 设置为守护线程，主程序结束时自动结束
    read_thread.start()

    # 测试通信
    print("\nSending signal...")
    arduino.write(b'T')
    time.sleep(1)

    # 测试夹爪控制
    print("\nOpening gripper...")
    arduino.write(b'O')
    time.sleep(2)

    print("\nChecking gripper status...")
    arduino.write(b'S')
    time.sleep(1)

    print("\nClosing gripper...")
    arduino.write(b'C')
    time.sleep(2)

    print("\nChecking gripper status...")
    arduino.write(b'S')
    time.sleep(1)

    print("\nReset gripper...")
    arduino.write(b'R')
    time.sleep(2)

    # 保持程序运行一段时间以便观察
    print("\nPress Ctrl+C exit...")
    while True:
        cmd = input("输入命令(O=打开, C=关闭, T=测试, S=状态, R=重置, Q=退出): ")
        if cmd.upper() == 'Q':
            break
        elif cmd.upper() in ['O', 'C', 'T', 'S', 'R']:
            arduino.write(cmd.upper().encode())
            time.sleep(1)

except KeyboardInterrupt:
    print("\nUser disconnected")
except Exception as e:
    print(f"error: {e}")
finally:
    try:
        arduino.close()
        print("\nConnection closed")
    except:
        pass