# app/models/user.py
from enum import Enum as pyEnum
from sqlalchemy import Column, Integer, LargeBinary, String, DateTime, Boolean, ForeignKey, Text, Enum as SQLEnum
from sqlalchemy.orm import relationship
from datetime import datetime, timedelta
from database import Base  # database.py에서 Base 클래스를 임포


class UserType(pyEnum):
    Standard = "Standard"
    Business = "Business"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    internal_id = Column(String, unique=True, index=True)  # 모든 유저에 대한 고유 식별자
    email = Column(String, index=True)
    name = Column(String)

    hashed_password = Column(String)  # 실제 저장되는 비밀번호는 해시됨

    user_type = Column(SQLEnum(UserType, name="user_type_enum"))

    is_additional_info_provided = Column(Boolean, default=False)

    auth_provider = Column(SQLEnum("Google", "Naver", "Kakao", "None", name="oauth_provier_enum"))
    auth_provider_id = Column(String, nullable=True)  # OAuth 공급자의 고유 식별자

    profile_picture_url = Column(String, nullable=True)  # 프로필 사진 URL
    embedded_profile = Column(LargeBinary, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    business_user = relationship("BusinessUser", back_populates="user", uselist=False)


class BusinessUser(Base):
    __tablename__ = "business_users"

    id = Column(Integer, ForeignKey("users.id"), primary_key=True)
    company_info = Column(Text)
    company_email = Column(String)
    company_address = Column(String)

    user = relationship("User", back_populates="business_user")


class EmailVerification(Base):
    __tablename__ = "email_verifications"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    verification_code = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_verified = Column(Boolean, default=False)  # 인증 여부

    def is_expired(self):
        return datetime.utcnow() > self.created_at + timedelta(minutes=3)


class PasswordResetVerification(Base):
    __tablename__ = "password_reset_verifications"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    verification_code = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_verified = Column(Boolean, default=False)  # 인증 여부

    def is_expired(self):
        return datetime.utcnow() > self.created_at + timedelta(minutes=3)
