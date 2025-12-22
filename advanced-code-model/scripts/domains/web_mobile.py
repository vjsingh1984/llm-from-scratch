"""
Web and Mobile Development Domain

Covers:
- React, Vue, Angular (Frontend)
- Node.js, FastAPI, Django (Backend)
- React Native, Flutter (Mobile)
- HTML/CSS, Tailwind
- REST APIs, GraphQL
"""

from typing import List, Optional
from .base import BaseDomain, DomainExample


class WebMobileDomain(BaseDomain):
    """Web and mobile development training examples."""

    def __init__(self, focus: Optional[str] = None):
        self.focus = focus

    def get_name(self) -> str:
        return "Web/Mobile Development"

    def get_description(self) -> str:
        return "Web and mobile development with React, Node.js, and modern frameworks"

    def get_subdomains(self) -> List[str]:
        return ["react", "nodejs", "fastapi", "react_native", "graphql"]

    def get_examples(self) -> List[DomainExample]:
        examples = []
        if not self.focus or self.focus == "frontend":
            examples.extend(self._react_examples())
        if not self.focus or self.focus == "backend":
            examples.extend(self._nodejs_examples())
            examples.extend(self._fastapi_examples())
        if not self.focus or self.focus == "mobile":
            examples.extend(self._react_native_examples())
        examples.extend(self._graphql_examples())
        return examples

    def _react_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Create a React component with TypeScript, hooks, and API integration",
                code='''import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';

// Types
interface User {
  id: string;
  name: string;
  email: string;
  role: 'admin' | 'user' | 'guest';
  createdAt: string;
}

interface UserFormData {
  name: string;
  email: string;
  role: User['role'];
}

interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page: number;
  pageSize: number;
  totalPages: number;
}

// API functions
const api = {
  getUsers: async (page: number, pageSize: number): Promise<PaginatedResponse<User>> => {
    const response = await fetch(`/api/users?page=${page}&pageSize=${pageSize}`);
    if (!response.ok) throw new Error('Failed to fetch users');
    return response.json();
  },

  createUser: async (data: UserFormData): Promise<User> => {
    const response = await fetch('/api/users', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (!response.ok) throw new Error('Failed to create user');
    return response.json();
  },

  deleteUser: async (id: string): Promise<void> => {
    const response = await fetch(`/api/users/${id}`, { method: 'DELETE' });
    if (!response.ok) throw new Error('Failed to delete user');
  },
};

// Custom hooks
function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedValue(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);

  return debouncedValue;
}

// Components
interface UserTableProps {
  users: User[];
  onDelete: (id: string) => void;
  isDeleting: boolean;
}

const UserTable: React.FC<UserTableProps> = ({ users, onDelete, isDeleting }) => {
  return (
    <table className="min-w-full divide-y divide-gray-200">
      <thead className="bg-gray-50">
        <tr>
          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
            Name
          </th>
          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
            Email
          </th>
          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
            Role
          </th>
          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
            Actions
          </th>
        </tr>
      </thead>
      <tbody className="bg-white divide-y divide-gray-200">
        {users.map((user) => (
          <tr key={user.id}>
            <td className="px-6 py-4 whitespace-nowrap">{user.name}</td>
            <td className="px-6 py-4 whitespace-nowrap">{user.email}</td>
            <td className="px-6 py-4 whitespace-nowrap">
              <span className={`px-2 py-1 text-xs rounded-full ${
                user.role === 'admin' ? 'bg-purple-100 text-purple-800' :
                user.role === 'user' ? 'bg-green-100 text-green-800' :
                'bg-gray-100 text-gray-800'
              }`}>
                {user.role}
              </span>
            </td>
            <td className="px-6 py-4 whitespace-nowrap">
              <button
                onClick={() => onDelete(user.id)}
                disabled={isDeleting}
                className="text-red-600 hover:text-red-900 disabled:opacity-50"
              >
                Delete
              </button>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
};

// Main component
export const UserManagement: React.FC = () => {
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState('');
  const [isFormOpen, setIsFormOpen] = useState(false);
  const pageSize = 10;

  const debouncedSearch = useDebounce(search, 300);
  const queryClient = useQueryClient();

  // Queries
  const { data, isLoading, error } = useQuery({
    queryKey: ['users', page, pageSize, debouncedSearch],
    queryFn: () => api.getUsers(page, pageSize),
    staleTime: 5000,
  });

  // Mutations
  const createMutation = useMutation({
    mutationFn: api.createUser,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users'] });
      setIsFormOpen(false);
    },
  });

  const deleteMutation = useMutation({
    mutationFn: api.deleteUser,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users'] });
    },
  });

  // Memoized filtered users
  const filteredUsers = useMemo(() => {
    if (!data?.data) return [];
    if (!debouncedSearch) return data.data;

    return data.data.filter(user =>
      user.name.toLowerCase().includes(debouncedSearch.toLowerCase()) ||
      user.email.toLowerCase().includes(debouncedSearch.toLowerCase())
    );
  }, [data?.data, debouncedSearch]);

  // Callbacks
  const handleDelete = useCallback((id: string) => {
    if (window.confirm('Are you sure you want to delete this user?')) {
      deleteMutation.mutate(id);
    }
  }, [deleteMutation]);

  const handleSubmit = useCallback((formData: UserFormData) => {
    createMutation.mutate(formData);
  }, [createMutation]);

  if (error) {
    return (
      <div className="p-4 bg-red-50 text-red-700 rounded-lg">
        Error loading users: {(error as Error).message}
      </div>
    );
  }

  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">User Management</h1>
        <button
          onClick={() => setIsFormOpen(true)}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          Add User
        </button>
      </div>

      <div className="mb-4">
        <input
          type="text"
          placeholder="Search users..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
        />
      </div>

      {isLoading ? (
        <div className="flex justify-center p-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600" />
        </div>
      ) : (
        <>
          <UserTable
            users={filteredUsers}
            onDelete={handleDelete}
            isDeleting={deleteMutation.isPending}
          />

          {data && (
            <div className="flex justify-between items-center mt-4">
              <span className="text-sm text-gray-700">
                Showing {(page - 1) * pageSize + 1} to {Math.min(page * pageSize, data.total)} of {data.total}
              </span>
              <div className="flex gap-2">
                <button
                  onClick={() => setPage(p => Math.max(1, p - 1))}
                  disabled={page === 1}
                  className="px-3 py-1 border rounded disabled:opacity-50"
                >
                  Previous
                </button>
                <button
                  onClick={() => setPage(p => p + 1)}
                  disabled={page >= data.totalPages}
                  className="px-3 py-1 border rounded disabled:opacity-50"
                >
                  Next
                </button>
              </div>
            </div>
          )}
        </>
      )}

      {isFormOpen && (
        <UserForm
          onSubmit={handleSubmit}
          onClose={() => setIsFormOpen(false)}
          isSubmitting={createMutation.isPending}
        />
      )}
    </div>
  );
};

// Form component
interface UserFormProps {
  onSubmit: (data: UserFormData) => void;
  onClose: () => void;
  isSubmitting: boolean;
}

const UserForm: React.FC<UserFormProps> = ({ onSubmit, onClose, isSubmitting }) => {
  const [formData, setFormData] = useState<UserFormData>({
    name: '',
    email: '',
    role: 'user',
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData);
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center">
      <div className="bg-white rounded-lg p-6 w-full max-w-md">
        <h2 className="text-xl font-bold mb-4">Add New User</h2>
        <form onSubmit={handleSubmit}>
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1">Name</label>
            <input
              type="text"
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              required
              className="w-full px-3 py-2 border rounded-lg"
            />
          </div>
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1">Email</label>
            <input
              type="email"
              value={formData.email}
              onChange={(e) => setFormData({ ...formData, email: e.target.value })}
              required
              className="w-full px-3 py-2 border rounded-lg"
            />
          </div>
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1">Role</label>
            <select
              value={formData.role}
              onChange={(e) => setFormData({ ...formData, role: e.target.value as User['role'] })}
              className="w-full px-3 py-2 border rounded-lg"
            >
              <option value="user">User</option>
              <option value="admin">Admin</option>
              <option value="guest">Guest</option>
            </select>
          </div>
          <div className="flex gap-2 justify-end">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 border rounded-lg"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSubmitting}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg disabled:opacity-50"
            >
              {isSubmitting ? 'Creating...' : 'Create'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default UserManagement;
''',
                domain="web",
                subdomain="react",
                tags=["react", "typescript", "hooks", "tanstack-query"],
                difficulty="advanced"
            ),
        ]

    def _nodejs_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Create an Express.js REST API with TypeScript, validation, and error handling",
                code='''import express, { Request, Response, NextFunction, Router } from 'express';
import { z } from 'zod';
import { PrismaClient } from '@prisma/client';
import jwt from 'jsonwebtoken';
import bcrypt from 'bcryptjs';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import { v4 as uuidv4 } from 'uuid';

// Initialize
const app = express();
const prisma = new PrismaClient();

// Middleware
app.use(helmet());
app.use(cors({ origin: process.env.CORS_ORIGIN }));
app.use(express.json());

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  message: { error: 'Too many requests, please try again later' },
});
app.use('/api', limiter);

// Types
interface AuthRequest extends Request {
  user?: { id: string; email: string; role: string };
}

// Validation schemas
const userSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8),
  name: z.string().min(2).max(100),
});

const loginSchema = z.object({
  email: z.string().email(),
  password: z.string(),
});

// Error handling
class AppError extends Error {
  constructor(
    public statusCode: number,
    public message: string,
    public isOperational = true
  ) {
    super(message);
    Error.captureStackTrace(this, this.constructor);
  }
}

// Validation middleware
const validate = (schema: z.ZodSchema) => {
  return (req: Request, res: Response, next: NextFunction) => {
    try {
      schema.parse(req.body);
      next();
    } catch (error) {
      if (error instanceof z.ZodError) {
        const errors = error.errors.map(e => ({
          field: e.path.join('.'),
          message: e.message,
        }));
        res.status(400).json({ error: 'Validation failed', details: errors });
      } else {
        next(error);
      }
    }
  };
};

// Auth middleware
const authenticate = async (
  req: AuthRequest,
  res: Response,
  next: NextFunction
) => {
  try {
    const authHeader = req.headers.authorization;
    if (!authHeader?.startsWith('Bearer ')) {
      throw new AppError(401, 'No token provided');
    }

    const token = authHeader.split(' ')[1];
    const decoded = jwt.verify(token, process.env.JWT_SECRET!) as {
      id: string;
      email: string;
      role: string;
    };

    const user = await prisma.user.findUnique({ where: { id: decoded.id } });
    if (!user) {
      throw new AppError(401, 'User not found');
    }

    req.user = { id: user.id, email: user.email, role: user.role };
    next();
  } catch (error) {
    if (error instanceof jwt.JsonWebTokenError) {
      next(new AppError(401, 'Invalid token'));
    } else {
      next(error);
    }
  }
};

// Authorization middleware
const authorize = (...roles: string[]) => {
  return (req: AuthRequest, res: Response, next: NextFunction) => {
    if (!req.user || !roles.includes(req.user.role)) {
      throw new AppError(403, 'Not authorized');
    }
    next();
  };
};

// Routes
const authRouter = Router();

authRouter.post('/register', validate(userSchema), async (req, res, next) => {
  try {
    const { email, password, name } = req.body;

    const existingUser = await prisma.user.findUnique({ where: { email } });
    if (existingUser) {
      throw new AppError(409, 'Email already registered');
    }

    const hashedPassword = await bcrypt.hash(password, 12);

    const user = await prisma.user.create({
      data: {
        id: uuidv4(),
        email,
        password: hashedPassword,
        name,
        role: 'user',
      },
      select: { id: true, email: true, name: true, role: true },
    });

    const token = jwt.sign(
      { id: user.id, email: user.email, role: user.role },
      process.env.JWT_SECRET!,
      { expiresIn: '7d' }
    );

    res.status(201).json({ user, token });
  } catch (error) {
    next(error);
  }
});

authRouter.post('/login', validate(loginSchema), async (req, res, next) => {
  try {
    const { email, password } = req.body;

    const user = await prisma.user.findUnique({ where: { email } });
    if (!user) {
      throw new AppError(401, 'Invalid credentials');
    }

    const isPasswordValid = await bcrypt.compare(password, user.password);
    if (!isPasswordValid) {
      throw new AppError(401, 'Invalid credentials');
    }

    const token = jwt.sign(
      { id: user.id, email: user.email, role: user.role },
      process.env.JWT_SECRET!,
      { expiresIn: '7d' }
    );

    res.json({
      user: { id: user.id, email: user.email, name: user.name, role: user.role },
      token,
    });
  } catch (error) {
    next(error);
  }
});

// User routes
const userRouter = Router();

userRouter.get('/', authenticate, authorize('admin'), async (req, res, next) => {
  try {
    const page = parseInt(req.query.page as string) || 1;
    const pageSize = parseInt(req.query.pageSize as string) || 10;
    const skip = (page - 1) * pageSize;

    const [users, total] = await Promise.all([
      prisma.user.findMany({
        skip,
        take: pageSize,
        select: { id: true, email: true, name: true, role: true, createdAt: true },
        orderBy: { createdAt: 'desc' },
      }),
      prisma.user.count(),
    ]);

    res.json({
      data: users,
      page,
      pageSize,
      total,
      totalPages: Math.ceil(total / pageSize),
    });
  } catch (error) {
    next(error);
  }
});

userRouter.get('/me', authenticate, async (req: AuthRequest, res, next) => {
  try {
    const user = await prisma.user.findUnique({
      where: { id: req.user!.id },
      select: { id: true, email: true, name: true, role: true, createdAt: true },
    });

    if (!user) {
      throw new AppError(404, 'User not found');
    }

    res.json(user);
  } catch (error) {
    next(error);
  }
});

userRouter.put('/me', authenticate, async (req: AuthRequest, res, next) => {
  try {
    const updateSchema = z.object({
      name: z.string().min(2).max(100).optional(),
      email: z.string().email().optional(),
    });

    const data = updateSchema.parse(req.body);

    const user = await prisma.user.update({
      where: { id: req.user!.id },
      data,
      select: { id: true, email: true, name: true, role: true },
    });

    res.json(user);
  } catch (error) {
    next(error);
  }
});

userRouter.delete('/:id', authenticate, authorize('admin'), async (req, res, next) => {
  try {
    await prisma.user.delete({ where: { id: req.params.id } });
    res.status(204).send();
  } catch (error) {
    next(error);
  }
});

// Mount routes
app.use('/api/auth', authRouter);
app.use('/api/users', userRouter);

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({ error: 'Not found' });
});

// Global error handler
app.use((err: Error, req: Request, res: Response, next: NextFunction) => {
  console.error(err);

  if (err instanceof AppError) {
    return res.status(err.statusCode).json({ error: err.message });
  }

  if (err instanceof z.ZodError) {
    return res.status(400).json({
      error: 'Validation failed',
      details: err.errors,
    });
  }

  res.status(500).json({ error: 'Internal server error' });
});

// Start server
const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

// Graceful shutdown
process.on('SIGTERM', async () => {
  console.log('SIGTERM received, shutting down...');
  await prisma.$disconnect();
  process.exit(0);
});

export default app;
''',
                domain="web",
                subdomain="nodejs",
                tags=["express", "typescript", "rest", "prisma"],
                difficulty="advanced"
            ),
        ]

    def _fastapi_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Create a FastAPI application with authentication and CRUD operations",
                code='''from fastapi import FastAPI, HTTPException, Depends, status, Query
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import create_engine, Column, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import uuid

# Configuration
SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Database setup
SQLALCHEMY_DATABASE_URL = "postgresql://user:password@localhost/dbname"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Models
class UserDB(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    name = Column(String)
    role = Column(String, default="user")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# Pydantic schemas
class UserBase(BaseModel):
    email: EmailStr
    name: str = Field(..., min_length=2, max_length=100)

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)

class UserUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=2, max_length=100)
    email: Optional[EmailStr] = None

class User(UserBase):
    id: str
    role: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    user_id: str

class PaginatedResponse(BaseModel):
    data: List[User]
    total: int
    page: int
    page_size: int
    total_pages: int

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Auth utilities
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> UserDB:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(UserDB).filter(UserDB.id == user_id).first()
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(
    current_user: UserDB = Depends(get_current_user)
) -> UserDB:
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

def require_role(*roles: str):
    async def role_checker(current_user: UserDB = Depends(get_current_active_user)):
        if current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions"
            )
        return current_user
    return role_checker

# FastAPI app
app = FastAPI(
    title="User API",
    description="REST API with authentication",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
@app.post("/token", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    user = db.query(UserDB).filter(UserDB.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.id}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/users", response_model=User, status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    existing_user = db.query(UserDB).filter(UserDB.email == user.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered"
        )

    db_user = UserDB(
        email=user.email,
        name=user.name,
        hashed_password=get_password_hash(user.password),
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.get("/users", response_model=PaginatedResponse)
async def list_users(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(require_role("admin"))
):
    skip = (page - 1) * page_size
    total = db.query(UserDB).count()
    users = db.query(UserDB).offset(skip).limit(page_size).all()

    return PaginatedResponse(
        data=users,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size,
    )

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: UserDB = Depends(get_current_active_user)):
    return current_user

@app.put("/users/me", response_model=User)
async def update_user_me(
    user_update: UserUpdate,
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(get_current_active_user)
):
    update_data = user_update.model_dump(exclude_unset=True)

    for field, value in update_data.items():
        setattr(current_user, field, value)

    db.commit()
    db.refresh(current_user)
    return current_user

@app.get("/users/{user_id}", response_model=User)
async def get_user(
    user_id: str,
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(require_role("admin"))
):
    user = db.query(UserDB).filter(UserDB.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: str,
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(require_role("admin"))
):
    user = db.query(UserDB).filter(UserDB.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    db.delete(user)
    db.commit()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
''',
                domain="web",
                subdomain="fastapi",
                tags=["fastapi", "python", "rest", "auth"],
                difficulty="advanced"
            ),
        ]

    def _react_native_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Create a React Native component with navigation and state management",
                code='''import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  FlatList,
  TouchableOpacity,
  StyleSheet,
  RefreshControl,
  ActivityIndicator,
  Image,
  Dimensions,
} from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import AsyncStorage from '@react-native-async-storage/async-storage';

// Types
type RootStackParamList = {
  Main: undefined;
  ProductDetail: { productId: string };
  Cart: undefined;
};

type TabParamList = {
  Home: undefined;
  Search: undefined;
  Profile: undefined;
};

interface Product {
  id: string;
  name: string;
  price: number;
  image: string;
  description: string;
  rating: number;
  reviews: number;
}

interface CartItem {
  product: Product;
  quantity: number;
}

// API
const api = {
  getProducts: async (): Promise<Product[]> => {
    const response = await fetch('https://api.example.com/products');
    if (!response.ok) throw new Error('Failed to fetch products');
    return response.json();
  },

  getProduct: async (id: string): Promise<Product> => {
    const response = await fetch(`https://api.example.com/products/${id}`);
    if (!response.ok) throw new Error('Failed to fetch product');
    return response.json();
  },
};

// Cart context
import { createContext, useContext, ReactNode } from 'react';

interface CartContextType {
  items: CartItem[];
  addToCart: (product: Product) => void;
  removeFromCart: (productId: string) => void;
  updateQuantity: (productId: string, quantity: number) => void;
  total: number;
}

const CartContext = createContext<CartContextType | undefined>(undefined);

export const CartProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [items, setItems] = useState<CartItem[]>([]);

  useEffect(() => {
    loadCart();
  }, []);

  useEffect(() => {
    saveCart();
  }, [items]);

  const loadCart = async () => {
    try {
      const saved = await AsyncStorage.getItem('cart');
      if (saved) setItems(JSON.parse(saved));
    } catch (error) {
      console.error('Failed to load cart:', error);
    }
  };

  const saveCart = async () => {
    try {
      await AsyncStorage.setItem('cart', JSON.stringify(items));
    } catch (error) {
      console.error('Failed to save cart:', error);
    }
  };

  const addToCart = (product: Product) => {
    setItems(current => {
      const existing = current.find(item => item.product.id === product.id);
      if (existing) {
        return current.map(item =>
          item.product.id === product.id
            ? { ...item, quantity: item.quantity + 1 }
            : item
        );
      }
      return [...current, { product, quantity: 1 }];
    });
  };

  const removeFromCart = (productId: string) => {
    setItems(current => current.filter(item => item.product.id !== productId));
  };

  const updateQuantity = (productId: string, quantity: number) => {
    if (quantity <= 0) {
      removeFromCart(productId);
      return;
    }
    setItems(current =>
      current.map(item =>
        item.product.id === productId ? { ...item, quantity } : item
      )
    );
  };

  const total = items.reduce(
    (sum, item) => sum + item.product.price * item.quantity,
    0
  );

  return (
    <CartContext.Provider
      value={{ items, addToCart, removeFromCart, updateQuantity, total }}
    >
      {children}
    </CartContext.Provider>
  );
};

const useCart = () => {
  const context = useContext(CartContext);
  if (!context) throw new Error('useCart must be used within CartProvider');
  return context;
};

// Components
const ProductCard: React.FC<{
  product: Product;
  onPress: () => void;
}> = ({ product, onPress }) => {
  const { addToCart } = useCart();

  return (
    <TouchableOpacity style={styles.card} onPress={onPress}>
      <Image source={{ uri: product.image }} style={styles.cardImage} />
      <View style={styles.cardContent}>
        <Text style={styles.cardTitle} numberOfLines={2}>
          {product.name}
        </Text>
        <View style={styles.ratingRow}>
          <Text style={styles.rating}>⭐ {product.rating.toFixed(1)}</Text>
          <Text style={styles.reviews}>({product.reviews})</Text>
        </View>
        <Text style={styles.price}>${product.price.toFixed(2)}</Text>
        <TouchableOpacity
          style={styles.addButton}
          onPress={() => addToCart(product)}
        >
          <Text style={styles.addButtonText}>Add to Cart</Text>
        </TouchableOpacity>
      </View>
    </TouchableOpacity>
  );
};

// Screens
const HomeScreen: React.FC<{ navigation: any }> = ({ navigation }) => {
  const [refreshing, setRefreshing] = useState(false);

  const { data: products, isLoading, error, refetch } = useQuery({
    queryKey: ['products'],
    queryFn: api.getProducts,
  });

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await refetch();
    setRefreshing(false);
  }, [refetch]);

  if (isLoading) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color="#007AFF" />
      </View>
    );
  }

  if (error) {
    return (
      <View style={styles.centered}>
        <Text style={styles.errorText}>Failed to load products</Text>
        <TouchableOpacity style={styles.retryButton} onPress={() => refetch()}>
          <Text style={styles.retryText}>Retry</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <FlatList
      data={products}
      numColumns={2}
      keyExtractor={item => item.id}
      contentContainerStyle={styles.listContainer}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
      }
      renderItem={({ item }) => (
        <ProductCard
          product={item}
          onPress={() =>
            navigation.navigate('ProductDetail', { productId: item.id })
          }
        />
      )}
    />
  );
};

const ProductDetailScreen: React.FC<{ route: any; navigation: any }> = ({
  route,
  navigation,
}) => {
  const { productId } = route.params;
  const { addToCart } = useCart();

  const { data: product, isLoading } = useQuery({
    queryKey: ['product', productId],
    queryFn: () => api.getProduct(productId),
  });

  if (isLoading || !product) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color="#007AFF" />
      </View>
    );
  }

  return (
    <View style={styles.detailContainer}>
      <Image source={{ uri: product.image }} style={styles.detailImage} />
      <View style={styles.detailContent}>
        <Text style={styles.detailTitle}>{product.name}</Text>
        <Text style={styles.detailPrice}>${product.price.toFixed(2)}</Text>
        <Text style={styles.detailDescription}>{product.description}</Text>
        <TouchableOpacity
          style={styles.buyButton}
          onPress={() => {
            addToCart(product);
            navigation.navigate('Cart');
          }}
        >
          <Text style={styles.buyButtonText}>Add to Cart</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const CartScreen: React.FC = () => {
  const { items, updateQuantity, removeFromCart, total } = useCart();

  if (items.length === 0) {
    return (
      <View style={styles.centered}>
        <Text style={styles.emptyText}>Your cart is empty</Text>
      </View>
    );
  }

  return (
    <View style={styles.cartContainer}>
      <FlatList
        data={items}
        keyExtractor={item => item.product.id}
        renderItem={({ item }) => (
          <View style={styles.cartItem}>
            <Image
              source={{ uri: item.product.image }}
              style={styles.cartItemImage}
            />
            <View style={styles.cartItemInfo}>
              <Text style={styles.cartItemName}>{item.product.name}</Text>
              <Text style={styles.cartItemPrice}>
                ${item.product.price.toFixed(2)}
              </Text>
              <View style={styles.quantityRow}>
                <TouchableOpacity
                  onPress={() =>
                    updateQuantity(item.product.id, item.quantity - 1)
                  }
                >
                  <Text style={styles.quantityButton}>-</Text>
                </TouchableOpacity>
                <Text style={styles.quantity}>{item.quantity}</Text>
                <TouchableOpacity
                  onPress={() =>
                    updateQuantity(item.product.id, item.quantity + 1)
                  }
                >
                  <Text style={styles.quantityButton}>+</Text>
                </TouchableOpacity>
              </View>
            </View>
            <TouchableOpacity onPress={() => removeFromCart(item.product.id)}>
              <Text style={styles.removeButton}>×</Text>
            </TouchableOpacity>
          </View>
        )}
      />
      <View style={styles.cartFooter}>
        <Text style={styles.totalText}>Total: ${total.toFixed(2)}</Text>
        <TouchableOpacity style={styles.checkoutButton}>
          <Text style={styles.checkoutButtonText}>Checkout</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

// Navigation
const Stack = createNativeStackNavigator<RootStackParamList>();
const Tab = createBottomTabNavigator<TabParamList>();

const MainTabs = () => (
  <Tab.Navigator>
    <Tab.Screen name="Home" component={HomeScreen} />
    <Tab.Screen name="Search" component={HomeScreen} />
    <Tab.Screen name="Profile" component={HomeScreen} />
  </Tab.Navigator>
);

export default function App() {
  return (
    <CartProvider>
      <NavigationContainer>
        <Stack.Navigator>
          <Stack.Screen
            name="Main"
            component={MainTabs}
            options={{ headerShown: false }}
          />
          <Stack.Screen name="ProductDetail" component={ProductDetailScreen} />
          <Stack.Screen name="Cart" component={CartScreen} />
        </Stack.Navigator>
      </NavigationContainer>
    </CartProvider>
  );
}

// Styles
const { width } = Dimensions.get('window');
const cardWidth = (width - 48) / 2;

const styles = StyleSheet.create({
  centered: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  listContainer: { padding: 16 },
  card: {
    width: cardWidth,
    backgroundColor: '#fff',
    borderRadius: 12,
    margin: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  cardImage: { width: '100%', height: 150, borderTopLeftRadius: 12, borderTopRightRadius: 12 },
  cardContent: { padding: 12 },
  cardTitle: { fontSize: 14, fontWeight: '600' },
  ratingRow: { flexDirection: 'row', alignItems: 'center', marginTop: 4 },
  rating: { fontSize: 12, color: '#FFB800' },
  reviews: { fontSize: 12, color: '#666', marginLeft: 4 },
  price: { fontSize: 18, fontWeight: 'bold', marginTop: 8 },
  addButton: { backgroundColor: '#007AFF', padding: 10, borderRadius: 8, marginTop: 8 },
  addButtonText: { color: '#fff', textAlign: 'center', fontWeight: '600' },
  detailContainer: { flex: 1, backgroundColor: '#fff' },
  detailImage: { width: '100%', height: 300 },
  detailContent: { padding: 16 },
  detailTitle: { fontSize: 24, fontWeight: 'bold' },
  detailPrice: { fontSize: 28, fontWeight: 'bold', color: '#007AFF', marginTop: 8 },
  detailDescription: { fontSize: 16, color: '#666', marginTop: 16, lineHeight: 24 },
  buyButton: { backgroundColor: '#007AFF', padding: 16, borderRadius: 12, marginTop: 24 },
  buyButtonText: { color: '#fff', textAlign: 'center', fontSize: 18, fontWeight: '600' },
  cartContainer: { flex: 1, backgroundColor: '#f5f5f5' },
  cartItem: { flexDirection: 'row', backgroundColor: '#fff', padding: 12, marginBottom: 1 },
  cartItemImage: { width: 80, height: 80, borderRadius: 8 },
  cartItemInfo: { flex: 1, marginLeft: 12 },
  cartItemName: { fontSize: 16, fontWeight: '500' },
  cartItemPrice: { fontSize: 14, color: '#007AFF', marginTop: 4 },
  quantityRow: { flexDirection: 'row', alignItems: 'center', marginTop: 8 },
  quantityButton: { fontSize: 20, paddingHorizontal: 12, color: '#007AFF' },
  quantity: { fontSize: 16, marginHorizontal: 8 },
  removeButton: { fontSize: 24, color: '#FF3B30', padding: 8 },
  cartFooter: { padding: 16, backgroundColor: '#fff', borderTopWidth: 1, borderTopColor: '#eee' },
  totalText: { fontSize: 20, fontWeight: 'bold', marginBottom: 12 },
  checkoutButton: { backgroundColor: '#34C759', padding: 16, borderRadius: 12 },
  checkoutButtonText: { color: '#fff', textAlign: 'center', fontSize: 18, fontWeight: '600' },
  errorText: { fontSize: 16, color: '#FF3B30' },
  retryButton: { marginTop: 12, padding: 12 },
  retryText: { color: '#007AFF', fontSize: 16 },
  emptyText: { fontSize: 18, color: '#666' },
});
''',
                domain="web",
                subdomain="react_native",
                tags=["react_native", "navigation", "context", "ecommerce"],
                difficulty="advanced"
            ),
        ]

    def _graphql_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Create a GraphQL schema and resolvers with authentication",
                code='''import { ApolloServer } from '@apollo/server';
import { expressMiddleware } from '@apollo/server/express4';
import { makeExecutableSchema } from '@graphql-tools/schema';
import express from 'express';
import cors from 'cors';
import jwt from 'jsonwebtoken';
import { GraphQLError } from 'graphql';
import DataLoader from 'dataloader';

// Types
interface User {
  id: string;
  email: string;
  name: string;
  role: 'admin' | 'user';
}

interface Product {
  id: string;
  name: string;
  price: number;
  categoryId: string;
}

interface Order {
  id: string;
  userId: string;
  items: OrderItem[];
  total: number;
  status: string;
  createdAt: Date;
}

interface OrderItem {
  productId: string;
  quantity: number;
  price: number;
}

interface Context {
  user?: User;
  dataSources: {
    userLoader: DataLoader<string, User>;
    productLoader: DataLoader<string, Product>;
  };
}

// Type definitions
const typeDefs = `#graphql
  type Query {
    me: User
    users: [User!]! @auth(requires: ADMIN)
    user(id: ID!): User @auth(requires: ADMIN)

    products(
      first: Int = 10
      after: String
      category: String
      minPrice: Float
      maxPrice: Float
    ): ProductConnection!
    product(id: ID!): Product

    orders: [Order!]! @auth
    order(id: ID!): Order @auth
  }

  type Mutation {
    login(email: String!, password: String!): AuthPayload!
    register(input: RegisterInput!): AuthPayload!

    createProduct(input: CreateProductInput!): Product! @auth(requires: ADMIN)
    updateProduct(id: ID!, input: UpdateProductInput!): Product! @auth(requires: ADMIN)
    deleteProduct(id: ID!): Boolean! @auth(requires: ADMIN)

    createOrder(input: CreateOrderInput!): Order! @auth
    updateOrderStatus(id: ID!, status: OrderStatus!): Order! @auth(requires: ADMIN)
  }

  type Subscription {
    orderStatusChanged(orderId: ID!): Order!
    productUpdated: Product!
  }

  # Directives
  directive @auth(requires: Role = USER) on FIELD_DEFINITION

  enum Role {
    ADMIN
    USER
  }

  # Types
  type User {
    id: ID!
    email: String!
    name: String!
    role: Role!
    orders: [Order!]!
    createdAt: DateTime!
  }

  type Product {
    id: ID!
    name: String!
    description: String
    price: Float!
    category: Category
    inventory: Int!
    images: [String!]!
    reviews: [Review!]!
    averageRating: Float
  }

  type Category {
    id: ID!
    name: String!
    products(first: Int = 10): [Product!]!
  }

  type Review {
    id: ID!
    user: User!
    product: Product!
    rating: Int!
    comment: String
    createdAt: DateTime!
  }

  type Order {
    id: ID!
    user: User!
    items: [OrderItem!]!
    total: Float!
    status: OrderStatus!
    shippingAddress: Address
    createdAt: DateTime!
    updatedAt: DateTime!
  }

  type OrderItem {
    product: Product!
    quantity: Int!
    price: Float!
  }

  type Address {
    street: String!
    city: String!
    state: String!
    country: String!
    postalCode: String!
  }

  enum OrderStatus {
    PENDING
    CONFIRMED
    PROCESSING
    SHIPPED
    DELIVERED
    CANCELLED
  }

  # Connections for pagination
  type ProductConnection {
    edges: [ProductEdge!]!
    pageInfo: PageInfo!
    totalCount: Int!
  }

  type ProductEdge {
    node: Product!
    cursor: String!
  }

  type PageInfo {
    hasNextPage: Boolean!
    hasPreviousPage: Boolean!
    startCursor: String
    endCursor: String
  }

  # Inputs
  input RegisterInput {
    email: String!
    password: String!
    name: String!
  }

  input CreateProductInput {
    name: String!
    description: String
    price: Float!
    categoryId: ID!
    inventory: Int!
    images: [String!]!
  }

  input UpdateProductInput {
    name: String
    description: String
    price: Float
    categoryId: ID
    inventory: Int
    images: [String!]
  }

  input CreateOrderInput {
    items: [OrderItemInput!]!
    shippingAddress: AddressInput!
  }

  input OrderItemInput {
    productId: ID!
    quantity: Int!
  }

  input AddressInput {
    street: String!
    city: String!
    state: String!
    country: String!
    postalCode: String!
  }

  # Payload types
  type AuthPayload {
    token: String!
    user: User!
  }

  scalar DateTime
`;

// Resolvers
const resolvers = {
  Query: {
    me: (_: unknown, __: unknown, context: Context) => {
      return context.user;
    },

    users: async (_: unknown, __: unknown, context: Context) => {
      // Database query
      return db.users.findAll();
    },

    products: async (
      _: unknown,
      args: { first: number; after?: string; category?: string }
    ) => {
      const { first, after, category } = args;

      let query = db.products.query();

      if (category) {
        query = query.where('categoryId', category);
      }

      if (after) {
        const cursor = decodeCursor(after);
        query = query.where('id', '>', cursor);
      }

      const products = await query.limit(first + 1).orderBy('id');
      const hasNextPage = products.length > first;
      const edges = products.slice(0, first).map((product: Product) => ({
        node: product,
        cursor: encodeCursor(product.id),
      }));

      return {
        edges,
        pageInfo: {
          hasNextPage,
          hasPreviousPage: !!after,
          startCursor: edges[0]?.cursor,
          endCursor: edges[edges.length - 1]?.cursor,
        },
        totalCount: await db.products.count(),
      };
    },

    product: async (_: unknown, { id }: { id: string }, context: Context) => {
      return context.dataSources.productLoader.load(id);
    },

    orders: async (_: unknown, __: unknown, context: Context) => {
      return db.orders.findByUserId(context.user!.id);
    },
  },

  Mutation: {
    login: async (_: unknown, { email, password }: { email: string; password: string }) => {
      const user = await db.users.findByEmail(email);
      if (!user || !await verifyPassword(password, user.passwordHash)) {
        throw new GraphQLError('Invalid credentials', {
          extensions: { code: 'UNAUTHENTICATED' },
        });
      }

      const token = jwt.sign(
        { userId: user.id, role: user.role },
        process.env.JWT_SECRET!,
        { expiresIn: '7d' }
      );

      return { token, user };
    },

    createProduct: async (
      _: unknown,
      { input }: { input: CreateProductInput },
      context: Context
    ) => {
      return db.products.create(input);
    },

    createOrder: async (
      _: unknown,
      { input }: { input: CreateOrderInput },
      context: Context
    ) => {
      // Validate products exist and have inventory
      const productIds = input.items.map(item => item.productId);
      const products = await context.dataSources.productLoader.loadMany(productIds);

      for (let i = 0; i < input.items.length; i++) {
        const product = products[i] as Product;
        const item = input.items[i];

        if (!product) {
          throw new GraphQLError(`Product ${item.productId} not found`);
        }
        if (product.inventory < item.quantity) {
          throw new GraphQLError(`Insufficient inventory for ${product.name}`);
        }
      }

      // Calculate total
      const items = input.items.map((item, i) => ({
        productId: item.productId,
        quantity: item.quantity,
        price: (products[i] as Product).price,
      }));

      const total = items.reduce(
        (sum, item) => sum + item.price * item.quantity,
        0
      );

      // Create order
      const order = await db.orders.create({
        userId: context.user!.id,
        items,
        total,
        status: 'PENDING',
        shippingAddress: input.shippingAddress,
      });

      // Update inventory
      for (const item of input.items) {
        await db.products.decrementInventory(item.productId, item.quantity);
      }

      return order;
    },
  },

  // Field resolvers
  User: {
    orders: (user: User) => db.orders.findByUserId(user.id),
  },

  Product: {
    category: (product: Product) => db.categories.findById(product.categoryId),
    reviews: (product: Product) => db.reviews.findByProductId(product.id),
    averageRating: async (product: Product) => {
      const reviews = await db.reviews.findByProductId(product.id);
      if (reviews.length === 0) return null;
      const sum = reviews.reduce((acc: number, r: { rating: number }) => acc + r.rating, 0);
      return sum / reviews.length;
    },
  },

  Order: {
    user: (order: Order, _: unknown, context: Context) =>
      context.dataSources.userLoader.load(order.userId),
    items: (order: Order) =>
      order.items.map((item: OrderItem) => ({
        ...item,
        product: () => db.products.findById(item.productId),
      })),
  },

  OrderItem: {
    product: (item: { productId: string }, _: unknown, context: Context) =>
      context.dataSources.productLoader.load(item.productId),
  },
};

// Auth directive
import { mapSchema, getDirective, MapperKind } from '@graphql-tools/utils';

function authDirectiveTransformer(schema: any) {
  return mapSchema(schema, {
    [MapperKind.OBJECT_FIELD]: (fieldConfig) => {
      const authDirective = getDirective(schema, fieldConfig, 'auth')?.[0];

      if (authDirective) {
        const { requires } = authDirective;
        const originalResolver = fieldConfig.resolve;

        fieldConfig.resolve = async (source, args, context, info) => {
          if (!context.user) {
            throw new GraphQLError('Not authenticated', {
              extensions: { code: 'UNAUTHENTICATED' },
            });
          }

          if (requires === 'ADMIN' && context.user.role !== 'admin') {
            throw new GraphQLError('Not authorized', {
              extensions: { code: 'FORBIDDEN' },
            });
          }

          return originalResolver?.(source, args, context, info);
        };
      }

      return fieldConfig;
    },
  });
}

// Server setup
async function startServer() {
  const app = express();

  let schema = makeExecutableSchema({ typeDefs, resolvers });
  schema = authDirectiveTransformer(schema);

  const server = new ApolloServer({
    schema,
    plugins: [
      {
        requestDidStart: async () => ({
          didResolveOperation: async ({ request, document }) => {
            console.log(`GraphQL operation: ${request.operationName}`);
          },
        }),
      },
    ],
  });

  await server.start();

  app.use(
    '/graphql',
    cors(),
    express.json(),
    expressMiddleware(server, {
      context: async ({ req }) => {
        // Get user from token
        let user: User | undefined;
        const token = req.headers.authorization?.replace('Bearer ', '');

        if (token) {
          try {
            const decoded = jwt.verify(token, process.env.JWT_SECRET!) as { userId: string };
            user = await db.users.findById(decoded.userId);
          } catch (error) {
            // Invalid token
          }
        }

        // Create data loaders
        const userLoader = new DataLoader(async (ids: readonly string[]) => {
          const users = await db.users.findByIds([...ids]);
          return ids.map(id => users.find((u: User) => u.id === id)!);
        });

        const productLoader = new DataLoader(async (ids: readonly string[]) => {
          const products = await db.products.findByIds([...ids]);
          return ids.map(id => products.find((p: Product) => p.id === id)!);
        });

        return {
          user,
          dataSources: { userLoader, productLoader },
        };
      },
    })
  );

  app.listen(4000, () => {
    console.log('Server running at http://localhost:4000/graphql');
  });
}

startServer();
''',
                domain="web",
                subdomain="graphql",
                tags=["graphql", "apollo", "typescript", "auth"],
                difficulty="advanced"
            ),
        ]
