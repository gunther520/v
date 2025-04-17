from mininet.net import Mininet
# Ensure Controller is imported
from mininet.node import Controller, OVSKernelSwitch
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import setLogLevel, info, error
import time
import sys
import os

### usage
#sudo /home/hkngae/anaconda3/envs/vllm/bin/python /home/hkngae/vllm/examples/offline_inference/mininet_test.py

# --- Set Log Level ---
setLogLevel('debug')

def run_simulation():
    info('*** Cleaning up previous Mininet runs (sudo mn -c)...\n')
    os.system('sudo mn -c')
    time.sleep(1)

    info('*** Creating network (Controller specified in constructor)\n')
    # Specify Controller ONLY in the constructor.
    # REMOVE build=False, let Mininet manage build/start order.
    net = Mininet()

    # DO NOT add the controller again manually
    # info('*** Adding controller instance manually\n')
    c0 = net.addController()

    info('*** Adding hosts\n')
    h1 = net.addHost('h1', ip='10.0.0.1/8')
    h2 = net.addHost('h2', ip='10.0.0.2/8')

    info('*** Adding switch\n')
    s1 = net.addSwitch('s1')

    info('*** Creating links\n')
    net.addLink(h1, s1,cls=TCLink,bw=4000)
    net.addLink(h2, s1)

    # DO NOT build manually
    # info('*** Building network topology\n')
    # net.build()

    info('*** Starting network (net.start() handles controller/switch start)\n')
    # Let net.start() handle starting the controller and switch
    # It SHOULD associate the controller passed in constructor with the switch
    try:
        net.start()
    except Exception as e:
        error(f"\n\n*** An error occurred during net.start(): {e}\n")
        import traceback
        traceback.print_exc()
        error("*** Exiting due to error during start.\n")
        # Try to cleanup even after error
        try:
            net.stop()
        except:
            pass
        return # Exit script

    # --- Check Controller Port AFTER net.start() ---
    if net.controllers:
        c0 = net.controllers[0]
        info(f"*** Controller '{c0.name}' should be listening on port {c0.port}.\n")
        info("*** Check 'sudo netstat -ntlp | grep LISTEN' in another terminal NOW.\n")
        os.system("sudo netstat -ntlp | grep LISTEN | grep ':%s '" % c0.port) # Check specific port
    else:
        error("!!! ERROR: No controller instance found AFTER net.start()!\n")
    # ----------------------------------------------

    # DO NOT start components manually
    # info('*** Starting controller\n')
    # c0.start()
    # info('*** Starting switch s1 and connecting to controller\n')
    # s1.start([c0])

    info('*** Network started. Waiting 5 seconds for components to settle...\n')
    time.sleep(5)

    # --- Check Switch Connection ---
    info('*** Checking switch connection to controller...\n')
    try:
        connected = s1.connected()
        info(f"Switch s1 connected state: {connected}\n")
        if not connected:
            info("*** Warning: Switch s1 reports not connected to controller!\n")
            try:
                controller_target = s1.cmd('ovs-vsctl get-controller s1')
                info(f"*** Switch s1 configured controller target: {controller_target.strip()}\n") # Did net.start() configure it?
                ovs_status = s1.cmd('ovs-vsctl show')
                info(f"*** OVS Status:\n{ovs_status}\n") # is_connected: true?
            except Exception as e_ovs:
                info(f"*** Could not query switch controller details: {e_ovs}\n")
        else:
             info("*** Switch s1 appears connected to controller.\n")
    except Exception as e_conn:
         error(f"*** Error checking switch connection: {e_conn}\n")
         # Fall through to check flows anyway

    # --- Check Flows ---
    info('*** Checking switch flows AFTER connection attempt...\n')
    try:
        time.sleep(1) # Give controller a moment
        flows = s1.dpctl('dump-flows')
        info(f"--- Flows on s1 ---\n{flows}\n-----------------\n")
        if not flows or ("actions=NORMAL" not in flows and "actions=CONTROLLER" not in flows):
             info("*** Warning: Standard L2/Controller flow rules MISSING on s1.\n")
        else:
             info("*** Standard L2/Controller flow rules FOUND on s1.\n")
    except Exception as e_flows:
        info(f"*** Warning: Could not execute dpctl dump-flows: {e_flows}\n")
    # -------------------

    # CLI
    #info('\n*** Pausing for manual check. Use the Mininet CLI ***\n')
    #CLI(net)

    # pingAll
    #info('*** Continuing script after CLI. Testing network connectivity (pingAll)...\n')
    results = net.pingAll()
    if results > 0:
       info(f"*** pingAll test failed with {results*100:.1f}% loss.\n")
    else:
       info("*** pingAll successful! ***\n")

    # --- Application Execution ---
    if results == 0: # Only run app if ping worked
        info('*** Running application test\n')
        # ... (Your VLLM application code - keep indented under the 'if') ...
        h1_ip = h1.IP()
        h2_ip = h2.IP()
        h1_iface = h1.intfNames()[0]
        h2_iface = h2.intfNames()[0]

        python_executable = '/home/hkngae/anaconda3/envs/vllm/bin/python'
        vllm_script = '/home/hkngae/vllm/examples/offline_inference/disaggregated_prefill_two_devices.py'
        server_log = '/tmp/h1_server_log.txt'

        h1.cmd(f'mkdir -p /tmp')
        h1.cmd(f'rm -f {server_log}')

        server_cmd = (f'export NCCL_DEBUG=DEBUG; '
                    f'export NCCL_SOCKET_IFNAME={h1_iface}; '
                    f'export HOST_IP={h1_ip}; '
                    f'export VLLM_HOST_IP={h1_ip}; '
                    f'export NCCL_P2P_DISABLE=1; '
                    f'export NCCL_SHM_DISABLE=1; '
                    f'export NCCL_IB_DISABLE=1; '
                    f'{python_executable} {vllm_script} --mode prefill --ip {h1_ip} '
                    f'> {server_log} 2>&1 &')

        info(f'Starting server on h1 (IP: {h1_ip}, Iface: {h1_iface})\n')
        info(f'Server logs will be in {server_log} on host h1\n')
        info(f'Server command:\n{server_cmd}\n')
        h1.cmd(server_cmd)

        initial_sleep_time = 10
        info(f'Waiting {initial_sleep_time} seconds for server to initialize...\n')
        time.sleep(initial_sleep_time)

        ps_output = h1.cmd('ps aux | grep disaggregated_prefill | grep -v grep')
        if vllm_script not in ps_output:
            info(f"*** Warning: Server process may not have started correctly on h1. Check {server_log}.\n")
            try:
                log_content = h1.cmd(f'cat {server_log}')
                info(f"--- Server Log ({server_log}) ---\n{log_content}\n----------------------------\n")
            except Exception as e_log:
                 info(f"Could not read server log: {e_log}\n")
        else:
            info("*** Server process appears to be running on h1.\n")

        client_cmd = (f'export NCCL_DEBUG=DEBUG; '
                    f'export NCCL_P2P_DISABLE=1; '
                    f'export NCCL_SHM_DISABLE=1; '
                    f'export NCCL_IB_DISABLE=1; '
                    f'export NCCL_SOCKET_IFNAME={h2_iface}; '
                    f'export HOST_IP={h2_ip}; '
                    f'export VLLM_HOST_IP={h2_ip}; '
                    f'{python_executable} {vllm_script} --mode decode --ip {h1_ip}')

        info(f'Running client on h2 (IP: {h2_ip}, Iface: {h2_iface})\n')
        info(f'Client command:\n{client_cmd}\n')
        info(f'--- Client Output Start ---\n')
        sys.stdout.flush()

        # Use net.terms += makeTerm(...) or h2.pexec(...) for interactive output if needed
        # Using waitOutput for simplicity here
        client_output = h2.cmd(client_cmd)
        print(client_output)
        sys.stdout.flush()


        info('\n--- Client Output End ---\n')
        info('Client command finished on h2.\n')

        info(f"--- Final Server Log ({server_log}) ---")
        try:
            log_content = h1.cmd(f'cat {server_log}')
            print(log_content)
        except Exception as e:
            info(f"Could not read server log: {e}")
        info(f"--------------------------------------\n")

    else:
        info("*** Skipping application test due to pingAll failure.\n")


    info('*** Stopping network')
    net.stop()

if __name__ == '__main__':
    setLogLevel('debug')
    run_simulation()
