FROM python:3.9

# Create a non-root user
RUN useradd -m -u 1000 user
USER user

# Set the environment variable to add the user's local bin to PATH
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory in the container
WORKDIR /app

# Copy and install dependencies
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the entire app to the container's working directory
COPY --chown=user . /app

# Run the application with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860","app:app"]

