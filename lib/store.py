from sqlalchemy import create_engine, Column, String, LargeBinary, select, delete
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
import pickle
import hashlib
from collections import OrderedDict
from langchain_core.stores import BaseStore
from langchain_core.documents.base import Document

Base = declarative_base()

class ByteStore(Base):
    __tablename__ = 'bytestore'
    collection_name = Column(String, primary_key=True)
    key = Column(String, primary_key=True)
    value = Column(LargeBinary)
    value_hash = Column(String)
    filename = Column(String, nullable=True)  
    
class PostgresByteStore(BaseStore):
    def __init__(self, conninfo, collection_name):
        self.conninfo = conninfo
        self.collection_name = collection_name

        self.engine = create_engine(conninfo)
        self.async_engine = create_async_engine(conninfo)

        Base.metadata.bind = self.engine
        Base.metadata.create_all(self.engine)

        self.Session = scoped_session(sessionmaker(bind=self.engine))
        self.async_session_factory = sessionmaker(self.async_engine, class_=AsyncSession, expire_on_commit=False)

    def compute_hash(self, content):
        hash_obj = hashlib.sha256(content.encode('utf-8'))
        return hash_obj.hexdigest()

    def serialize_value(self, value):
        return pickle.dumps(self.recursive_ordered_dict(value))

    def recursive_ordered_dict(self, obj):
        if isinstance(obj, dict):
            return OrderedDict((k, self.recursive_ordered_dict(v)) for k, v in sorted(obj.items()))
        elif isinstance(obj, list):
            return [self.recursive_ordered_dict(v) for v in obj]
        else:
            return obj

    def extract_hashable_content(self, value):
        if isinstance(value, Document):
            return value.page_content
        elif isinstance(value, dict):
            return value.get('page_content', '')
        else:
            return str(value)

    def get(self, key):
        with self.Session() as session:
            result = session.execute(select(ByteStore).filter_by(collection_name=self.collection_name, key=key)).scalar()
            return pickle.loads(result.value) if result else None

    def set(self, key, value, filename=None):
        with self.Session() as session:
            serialized_value = self.serialize_value(value)
            hashable_content = self.extract_hashable_content(value)
            entry = ByteStore(
                collection_name=self.collection_name,
                key=key,
                value=serialized_value,
                value_hash=self.compute_hash(hashable_content),
                filename=filename  
            )
            session.merge(entry)
            session.commit()

    def mget(self, keys):
        results = {}
        with self.Session() as session:
            query_results = session.execute(select(ByteStore).where(ByteStore.collection_name == self.collection_name, ByteStore.key.in_(keys))).scalars()
            for result in query_results:
                results[result.key] = pickle.loads(result.value)
        return [results.get(key) for key in keys]

    def mset(self, items):
        with self.Session() as session:
            for key, value, filename in items:
                serialized_value = self.serialize_value(value)
                hashable_content = self.extract_hashable_content(value)
                entry = ByteStore(
                    collection_name=self.collection_name,
                    key=key,
                    value=serialized_value,
                    value_hash=self.compute_hash(hashable_content),
                    filename=filename  
                )
                session.merge(entry)
            session.commit()

    def mdelete(self, keys):
        with self.Session() as session:
            session.execute(delete(ByteStore).where(ByteStore.collection_name == self.collection_name, ByteStore.key.in_(keys)))
            session.commit()

    def yield_keys(self, prefix=None):
        with self.Session() as session:
            query = select(ByteStore.key).where(ByteStore.collection_name == self.collection_name)
            if prefix:
                query = query.where(ByteStore.key.like(f'{prefix}%'))
            for row in session.execute(query):
                yield row.key

    # Async methods

    async def aset(self, key, value, filename=None):
        async with self.async_session_factory() as session:
            serialized_value = self.serialize_value(value)
            hashable_content = self.extract_hashable_content(value)
            entry = ByteStore(
                collection_name=self.collection_name,
                key=key,
                value=serialized_value,
                value_hash=self.compute_hash(hashable_content),
                filename=filename  
            )
            session.merge(entry)
            await session.commit()

    async def amset(self, items):
        async with self.async_session_factory() as session:
            for key, value, filename in items:
                serialized_value = self.serialize_value(value)
                hashable_content = self.extract_hashable_content(value)
                entry = ByteStore(
                    collection_name=self.collection_name,
                    key=key,
                    value=serialized_value,
                    value_hash=self.compute_hash(hashable_content),
                    filename=filename  
                )
                session.merge(entry)
            await session.commit()

    async def aget(self, key):
        async with self.async_session_factory() as session:
            result = await session.execute(select(ByteStore).filter_by(collection_name=self.collection_name, key=key))
            byte_store = result.scalars().first()
            return pickle.loads(byte_store.value) if byte_store else None

    async def amget(self, keys):
        results = {}
        async with self.async_session_factory() as session:
            query_results = await session.execute(select(ByteStore).where(ByteStore.collection_name == self.collection_name, ByteStore.key.in_(keys)))
            for result in query_results.scalars():
                results[result.key] = pickle.loads(result.value)
        return [results.get(key) for key in keys]

    async def amdelete(self, keys):
        async with self.async_session_factory() as session:
            await session.execute(delete(ByteStore).where(ByteStore.collection_name == self.collection_name, ByteStore.key.in_(keys)))
            await session.commit()

    async def ayield_keys(self, prefix=None):
        async with self.async_session_factory() as session:
            query = select(ByteStore.key).where(ByteStore.collection_name == self.collection_name)
            if prefix:
                query = query.where(ByteStore.key.like(f'{prefix}%'))
            async for row in session.stream(query):
                yield row.key
