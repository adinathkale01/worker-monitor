# First download the yolov8n-pose.pt from following link

https://docs.ultralytics.com/models/yolov8/

# Export to engine file using following command 

#### yolo export model=yolov8n-pose.pt format=engine half=True device=0

# Startup Service: Configured with system-level service (e.g., systemd) for auto-start.
 ### 1.Create your systemd service file
   #### worker_monitor.service
 
 ### 2.Move the service file to systemd
   #### sudo cp worker_monitor.service /etc/systemd/system/
 ### 3.Reload systemd and enable your service

  #### sudo systemctl daemon-reexec
  #### sudo systemctl daemon-reload
  #### sudo systemctl enable worker_monitor.service
  #### sudo systemctl start worker_monitor.service
  
 ### 4. To view Logs & Debugging
  #### journalctl -u worker_monitor.service -f

### Once we done the steps above and enabled the service, script will automatically run every time the system boots up


