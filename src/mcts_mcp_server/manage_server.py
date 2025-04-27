#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCTS Server Manager
===================

This script provides utilities to start, stop, and check the status of
the MCTS MCP server to ensure only one instance is running at a time.
"""
import os
import sys
import argparse
import signal
import time
import subprocess
import psutil

def find_server_process():
    """Find the running MCTS server process if it exists."""
    current_pid = os.getpid()  # Get this script's PID to avoid self-identification
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Skip this process
            if proc.pid == current_pid:
                continue
                
            cmdline = proc.info.get('cmdline', [])
            cmdline_str = ' '.join(cmdline) if cmdline else ''
            
            # Check for server.py but not manage_server.py
            if ('server.py' in cmdline_str and 
                'python' in cmdline_str and 
                'manage_server.py' not in cmdline_str):
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None

def start_server():
    """Start the MCTS server if it's not already running."""
    proc = find_server_process()
    if proc:
        print(f"MCTS server is already running with PID {proc.pid}")
        return False

    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Start the server using subprocess
    try:
        # Use nohup to keep the server running after this script exits
        cmd = f"cd {script_dir} && python -u server.py > {script_dir}/server.log 2>&1"
        subprocess.Popen(cmd, shell=True, start_new_session=True)
        print("MCTS server started successfully")
        
        # Wait a moment to verify it started
        time.sleep(2)
        proc = find_server_process()
        if proc:
            print(f"Server process running with PID {proc.pid}")
            return True
        else:
            print("Server process not found after startup. Check server.log for errors.")
            return False
    except Exception as e:
        print(f"Error starting server: {e}")
        return False

def stop_server():
    """Stop the MCTS server if it's running."""
    proc = find_server_process()
    if not proc:
        print("MCTS server is not running")
        return True
    
    try:
        # Try to terminate gracefully first
        proc.send_signal(signal.SIGTERM)
        print(f"Sent SIGTERM to process {proc.pid}")
        
        # Wait up to 5 seconds for process to terminate
        for i in range(5):
            if not psutil.pid_exists(proc.pid):
                print("Server stopped successfully")
                return True
            time.sleep(1)
        
        # If still running, force kill
        if psutil.pid_exists(proc.pid):
            proc.send_signal(signal.SIGKILL)
            print(f"Force killed process {proc.pid}")
            time.sleep(1)
            
        if not psutil.pid_exists(proc.pid):
            print("Server stopped successfully")
            return True
        else:
            print("Failed to stop server")
            return False
    except Exception as e:
        print(f"Error stopping server: {e}")
        return False

def check_status():
    """Check the status of the MCTS server."""
    proc = find_server_process()
    if proc:
        print(f"MCTS server is running with PID {proc.pid}")
        # Get the uptime
        try:
            create_time = proc.create_time()
            uptime = time.time() - create_time
            hours, remainder = divmod(uptime, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"Server uptime: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            print("Unable to determine server uptime")
        return True
    else:
        print("MCTS server is not running")
        return False

def restart_server():
    """Restart the MCTS server."""
    stop_server()
    # Wait a moment to ensure resources are released
    time.sleep(2)
    return start_server()

def main():
    """Parse arguments and execute the appropriate command."""
    parser = argparse.ArgumentParser(description="Manage the MCTS server")
    parser.add_argument('command', choices=['start', 'stop', 'restart', 'status'],
                       help='Command to execute')
    
    args = parser.parse_args()
    
    if args.command == 'start':
        start_server()
    elif args.command == 'stop':
        stop_server()
    elif args.command == 'restart':
        restart_server()
    elif args.command == 'status':
        check_status()

if __name__ == "__main__":
    main()
