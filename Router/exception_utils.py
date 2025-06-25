import traceback
from datetime import datetime

# Utility function to log exceptions to the database
def log_exception(error, function_name=None, context=None, session=None, user_id=None, db=None):
    """
    Logs an exception to the ExceptionLog table.
    Args:
        error: Exception object or string
        function_name: Name of the function where the error occurred
        context: Additional context about the error
        session: SQLAlchemy session object (legacy parameter)
        user_id: (Optional) ID of the user related to the error
        db: SQLAlchemy session object (alternative parameter name)
    """
    # Handle both 'session' and 'db' parameter names for backwards compatibility
    db_session = session or db
    
    if db_session:
        try:
            # Import here to avoid circular imports
            from .relations import ExceptionLog
            
            error_message = str(error)
            stack_trace = traceback.format_exc()
            
            # Create exception log entry
            exception_log = ExceptionLog(
                error_message=error_message,
                stack_trace=stack_trace,
                user_id=user_id,
                created_at=datetime.utcnow()
            )
            
            db_session.add(exception_log)
            db_session.commit()
            
            print(f"✅ Exception logged to database: {function_name} - {error_message[:100]}...")
            
        except Exception as db_error:
            # If database logging fails, rollback and print to console
            try:
                db_session.rollback()
            except:
                pass
            print(f"❌ Failed to log exception to database: {str(db_error)}")
            print(f"Original error in {function_name}: {str(error)}")
            if context:
                print(f"Context: {context}")
            print(traceback.format_exc())
    else:
        # If no session is provided, just print the error
        print(f"⚠️  No database session provided for exception logging")
        print(f"Error in {function_name}: {str(error)}")
        if context:
            print(f"Context: {context}")
        print(traceback.format_exc()) 