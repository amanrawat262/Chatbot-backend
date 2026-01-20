from sqlalchemy import create_engine, MetaData
# from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base

Base = declarative_base()
DATABASE_URL = "postgresql+psycopg2://postgres:Esya%401234@10.159.1.39:5432/aiml_test"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

