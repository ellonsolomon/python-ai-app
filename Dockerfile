# Start with the TensorFlow base image  
FROM tensorflow/tensorflow:latest  
  
# Set working directory for our Flask + Dash application  
WORKDIR /usr/src/app  
  
# Install virtualenv  
RUN pip install virtualenv  
  
# Create a virtual environment and activate it  
RUN virtualenv venv  
ENV VIRTUAL_ENV /usr/src/app/venv  
ENV PATH /usr/src/app/venv/bin:$PATH  
  
# Copy both backend and frontend application files into the container  
COPY . .  
  
# Install backend dependencies in the virtual environment  
RUN pip install --no-cache-dir -r requirements.txt  
  
# Install additional dependencies for Dash frontend  
RUN pip install dash pandas plotly  
  
# Expose the port that both the backend and frontend will use  
EXPOSE 8050  
  
# Run the frontend Dash application on the specified port  
CMD ["python", "./app_lab3_template.py"]  
