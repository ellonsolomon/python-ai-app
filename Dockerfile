# Start with a Python 3.10 base image  
FROM python:3.10.12-slim  

# Set working directory for the application (Flask + Dash)  
WORKDIR /usr/src/app

# Copy both backend and frontend application files into the container  
COPY . .  

# Install backend dependencies in the virtual environment  
# RUN pip install --no-cache-dir -r requirements.txt  

# Install TensorFlow 2.16.1 specifically  
RUN pip install tensorflow==2.16.1  

# Install additional dependencies for Dash frontend  
RUN pip install dash pandas plotly flask numpy scikit-learn

# Copy both backend and frontend application files and the start script into the container  
COPY . .  
COPY start_services.sh .  

# Make the start_services.sh script executable  
RUN chmod +x ./start_services.sh  

# Expose the ports used by the frontend and backend services  
EXPOSE 8050 4000  

# Run the script to start both services  
CMD ["./start_services.sh"]  