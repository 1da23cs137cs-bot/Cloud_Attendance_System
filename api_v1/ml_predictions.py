import tensorflow as tf
import numpy as np
from .models import Attendance, Student

def predict_who_might_skip_class(student_id):
    """
    This function looks at a student's attendance history
    and predicts if they might skip class.
    It's like fortune telling but with math!
    """
    student = Student.objects.get(id=student_id)
    
    # Get all their attendance records
    attendance_records = Attendance.objects.filter(
        student=student
    ).order_by('date').values_list('status', flat=True)
    
    # If they don't have enough history, we can't predict
    if len(attendance_records) < 10:
        return "Not enough data yet!"
    
    # Turn attendance into numbers the AI can understand
    # P (Present) = 1 (good)
    # L (Late) = 0.5 (okay)
    # E (Excused) = 0 (neutral)
    # A (Absent) = -1 (bad)
    mapping = {'P': 1, 'L': 0.5, 'E': 0, 'A': -1}
    numbers = np.array([mapping.get(s, 0) for s in attendance_records[-30:]])
    
    # Create a simple AI brain
    brain = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(30,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    brain.compile(optimizer='adam', loss='binary_crossentropy')
    
    # Make the prediction
    risk_score = brain.predict(numbers.reshape(1, -1))[0][0]
    
    if risk_score > 0.7:
        return "🔴 High risk - This student might skip class!"
    elif risk_score > 0.4:
        return "🟡 Medium risk - Watch this student"
    else:
        return "🟢 Low risk - Good student!"