import paramiko

def run_kill():
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        print("Connecting to Gateway...")
        client.connect('172.16.13.62', port=22, username='joseva', password='sastra@2026')
        
        print("Killing remote_notifier on gateway...")
        stdin, stdout, stderr = client.exec_command("pkill -f remote_notifier")
        print("Gateway out:", stdout.read().decode())
        print("Gateway err:", stderr.read().decode())

        print("Killing remote_notifier on node1...")
        # since ssh node1 requires no password if keys are setup, let's try
        stdin, stdout, stderr = client.exec_command("ssh node1 pkill -f remote_notifier")
        print("node1 out:", stdout.read().decode())
        print("node1 err:", stderr.read().decode())
        
        client.close()
        print("Done.")
    except Exception as e:
        print("Error:", e)

if __name__ == '__main__':
    run_kill()
