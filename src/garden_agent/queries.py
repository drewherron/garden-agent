"""Optimized database queries and performance utilities."""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session, joinedload, selectinload
from sqlalchemy import func, desc, asc, and_, or_, text, case
from sqlalchemy.sql import label

from .models import Plant, Planting, Conversation, Message, KnowledgeBase, Season
from .database import db_manager


class PlantQueries:
    """Optimized queries for plant-related operations."""
    
    @staticmethod
    def get_plants_with_planting_counts(
        session: Session,
        year: Optional[int] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get plants with their planting counts (optimized with joins)."""
        query = session.query(
            Plant.id,
            Plant.name,
            Plant.sun_requirement,
            Plant.soil_ph_min,
            Plant.soil_ph_max,
            func.count(Planting.id).label('total_plantings'),
            func.max(Planting.created_at).label('last_planted')
        ).outerjoin(Planting, Plant.id == Planting.plant_id)
        
        if year:
            query = query.filter(
                or_(Planting.year == year, Planting.year.is_(None))
            )
        
        results = query.group_by(
            Plant.id, Plant.name, Plant.sun_requirement, 
            Plant.soil_ph_min, Plant.soil_ph_max
        ).order_by(desc('total_plantings')).limit(limit).all()
        
        return [
            {
                'id': r.id,
                'name': r.name,
                'sun_requirement': r.sun_requirement,
                'soil_ph_min': r.soil_ph_min,
                'soil_ph_max': r.soil_ph_max,
                'total_plantings': r.total_plantings,
                'last_planted': r.last_planted
            }
            for r in results
        ]
    
    @staticmethod
    def get_similar_plants_by_requirements(
        session: Session,
        reference_plant_id: int,
        limit: int = 10
    ) -> List[Plant]:
        """Find plants with similar growing requirements."""
        # Get reference plant
        ref_plant = session.query(Plant).filter(Plant.id == reference_plant_id).first()
        if not ref_plant:
            return []
        
        # Find similar plants
        query = session.query(Plant).filter(
            and_(
                Plant.id != reference_plant_id,
                Plant.sun_requirement == ref_plant.sun_requirement
            )
        )
        
        # Add pH range similarity if available
        if ref_plant.soil_ph_min and ref_plant.soil_ph_max:
            ph_tolerance = 0.5
            query = query.filter(
                and_(
                    Plant.soil_ph_min <= (ref_plant.soil_ph_max + ph_tolerance),
                    Plant.soil_ph_max >= (ref_plant.soil_ph_min - ph_tolerance)
                )
            )
        
        return query.limit(limit).all()


class PlantingQueries:
    """Optimized queries for planting operations."""
    
    @staticmethod
    def get_active_plantings_with_plants(
        session: Session,
        year: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get active plantings with plant info (single query with join)."""
        current_year = year or datetime.now().year
        
        results = session.query(
            Planting.id,
            Planting.year,
            Planting.season,
            Planting.batch_number,
            Planting.seed_start_date,
            Planting.transplant_date,
            Planting.notes,
            Plant.name.label('plant_name'),
            Plant.sun_requirement,
            Plant.maturity_days
        ).join(Plant, Planting.plant_id == Plant.id).filter(
            Planting.year == current_year
        ).order_by(desc(Planting.created_at)).all()
        
        return [
            {
                'id': r.id,
                'year': r.year,
                'season': r.season,
                'batch_number': r.batch_number,
                'seed_start_date': r.seed_start_date,
                'transplant_date': r.transplant_date,
                'notes': r.notes,
                'plant_name': r.plant_name,
                'sun_requirement': r.sun_requirement,
                'maturity_days': r.maturity_days
            }
            for r in results
        ]
    
    @staticmethod
    def get_planting_timeline(
        session: Session,
        year: int,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get chronological planting timeline."""
        # Create a unified timeline from seed_start_date and transplant_date
        seed_events = session.query(
            Planting.id,
            Planting.seed_start_date.label('event_date'),
            label('event_type', 'seed_start'),
            Plant.name.label('plant_name'),
            Planting.season,
            Planting.batch_number
        ).join(Plant).filter(
            and_(
                Planting.year == year,
                Planting.seed_start_date.is_not(None)
            )
        )
        
        transplant_events = session.query(
            Planting.id,
            Planting.transplant_date.label('event_date'),
            label('event_type', 'transplant'),
            Plant.name.label('plant_name'),
            Planting.season,
            Planting.batch_number
        ).join(Plant).filter(
            and_(
                Planting.year == year,
                Planting.transplant_date.is_not(None)
            )
        )
        
        # Union and order by date
        combined = seed_events.union(transplant_events).order_by(
            'event_date'
        ).limit(limit).all()
        
        return [
            {
                'planting_id': r.id,
                'event_date': r.event_date,
                'event_type': r.event_type,
                'plant_name': r.plant_name,
                'season': r.season,
                'batch_number': r.batch_number
            }
            for r in combined
        ]
    
    @staticmethod
    def get_seasonal_statistics(
        session: Session,
        year: int
    ) -> Dict[str, Any]:
        """Get comprehensive seasonal planting statistics."""
        # Base query for the year
        base_query = session.query(Planting).filter(Planting.year == year)
        
        # Seasonal breakdown
        seasonal_counts = session.query(
            Planting.season,
            func.count(Planting.id).label('count'),
            func.count(func.distinct(Planting.plant_id)).label('unique_plants')
        ).filter(Planting.year == year).group_by(Planting.season).all()
        
        # Monthly breakdown (from seed_start_date and transplant_date)
        monthly_seeds = session.query(
            func.strftime('%m', Planting.seed_start_date).label('month'),
            func.count(Planting.id).label('count')
        ).filter(
            and_(
                Planting.year == year,
                Planting.seed_start_date.is_not(None)
            )
        ).group_by('month').all()
        
        monthly_transplants = session.query(
            func.strftime('%m', Planting.transplant_date).label('month'),
            func.count(Planting.id).label('count')
        ).filter(
            and_(
                Planting.year == year,
                Planting.transplant_date.is_not(None)
            )
        ).group_by('month').all()
        
        return {
            'year': year,
            'seasonal_breakdown': [
                {
                    'season': s.season,
                    'total_plantings': s.count,
                    'unique_plants': s.unique_plants
                }
                for s in seasonal_counts
            ],
            'monthly_seed_starts': {m.month: m.count for m in monthly_seeds},
            'monthly_transplants': {m.month: m.count for m in monthly_transplants},
            'total_plantings': base_query.count()
        }


class ConversationQueries:
    """Optimized queries for conversation operations."""
    
    @staticmethod
    def get_conversations_with_message_counts(
        session: Session,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get conversations with message counts (single query)."""
        results = session.query(
            Conversation.id,
            Conversation.title,
            Conversation.created_at,
            Conversation.updated_at,
            func.count(Message.id).label('message_count'),
            func.max(Message.created_at).label('last_message_at')
        ).outerjoin(Message).group_by(
            Conversation.id, Conversation.title, 
            Conversation.created_at, Conversation.updated_at
        ).order_by(desc('last_message_at')).limit(limit).all()
        
        return [
            {
                'id': r.id,
                'title': r.title,
                'created_at': r.created_at,
                'updated_at': r.updated_at,
                'message_count': r.message_count,
                'last_message_at': r.last_message_at
            }
            for r in results
        ]
    
    @staticmethod
    def search_conversations_by_content(
        session: Session,
        search_term: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search conversations by message content."""
        # Find conversations that contain messages with the search term
        results = session.query(
            Conversation.id,
            Conversation.title,
            Conversation.updated_at,
            func.count(Message.id).label('matching_messages'),
            func.group_concat(Message.content, ' | ').label('message_preview')
        ).join(Message).filter(
            Message.content.ilike(f"%{search_term}%")
        ).group_by(
            Conversation.id, Conversation.title, Conversation.updated_at
        ).order_by(desc('matching_messages')).limit(limit).all()
        
        return [
            {
                'conversation_id': r.id,
                'title': r.title,
                'updated_at': r.updated_at,
                'matching_messages': r.matching_messages,
                'preview': r.message_preview[:200] + '...' if len(r.message_preview) > 200 else r.message_preview
            }
            for r in results
        ]


class KnowledgeQueries:
    """Optimized queries for knowledge base operations."""
    
    @staticmethod
    def get_documents_with_chunk_counts(
        session: Session,
        category: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get knowledge documents with chunk counts."""
        from .models import KnowledgeChunk
        
        query = session.query(
            KnowledgeBase.id,
            KnowledgeBase.title,
            KnowledgeBase.category,
            KnowledgeBase.source,
            KnowledgeBase.created_at,
            func.count(KnowledgeChunk.id).label('chunk_count'),
            func.sum(KnowledgeChunk.token_count).label('total_tokens')
        ).outerjoin(KnowledgeChunk).group_by(
            KnowledgeBase.id, KnowledgeBase.title, KnowledgeBase.category,
            KnowledgeBase.source, KnowledgeBase.created_at
        )
        
        if category:
            query = query.filter(KnowledgeBase.category == category)
        
        results = query.order_by(desc('total_tokens')).limit(limit).all()
        
        return [
            {
                'id': r.id,
                'title': r.title,
                'category': r.category,
                'source': r.source,
                'created_at': r.created_at,
                'chunk_count': r.chunk_count,
                'total_tokens': r.total_tokens or 0
            }
            for r in results
        ]
    
    @staticmethod
    def get_category_statistics(session: Session) -> List[Dict[str, Any]]:
        """Get knowledge base category statistics."""
        results = session.query(
            KnowledgeBase.category,
            func.count(KnowledgeBase.id).label('document_count'),
            func.avg(func.length(KnowledgeBase.content)).label('avg_length')
        ).filter(
            KnowledgeBase.category.is_not(None)
        ).group_by(KnowledgeBase.category).order_by(
            desc('document_count')
        ).all()
        
        return [
            {
                'category': r.category,
                'document_count': r.document_count,
                'average_length': int(r.avg_length) if r.avg_length else 0
            }
            for r in results
        ]


class PerformanceQueries:
    """Performance monitoring and optimization queries."""
    
    @staticmethod
    def analyze_query_performance(session: Session) -> Dict[str, Any]:
        """Analyze database performance metrics."""
        # SQLite specific performance queries
        try:
            # Check database size
            size_result = session.execute(text("""
                SELECT page_count * page_size as size 
                FROM pragma_page_count(), pragma_page_size()
            """)).fetchone()
            
            # Check table sizes
            table_sizes = session.execute(text("""
                SELECT name, COUNT(*) as row_count
                FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """)).fetchall()
            
            # Check index usage
            indexes = session.execute(text("""
                SELECT name, tbl_name, sql 
                FROM sqlite_master 
                WHERE type='index' AND sql IS NOT NULL
            """)).fetchall()
            
            return {
                'database_size_bytes': size_result[0] if size_result else 0,
                'table_info': [
                    {'table': name, 'rows': count} 
                    for name, count in table_sizes
                ],
                'indexes': [
                    {'name': name, 'table': table, 'sql': sql}
                    for name, table, sql in indexes
                ]
            }
        
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def get_slow_query_suggestions(session: Session) -> List[str]:
        """Get suggestions for query optimization."""
        suggestions = []
        
        # Check for tables without indexes
        try:
            tables_without_indexes = session.execute(text("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                AND name NOT IN (
                    SELECT DISTINCT tbl_name FROM sqlite_master 
                    WHERE type='index' AND sql IS NOT NULL
                )
            """)).fetchall()
            
            for (table,) in tables_without_indexes:
                suggestions.append(f"Consider adding indexes to table '{table}'")
            
            # Check for large tables
            large_tables = session.execute(text("""
                SELECT name FROM sqlite_master m, pragma_table_info(m.name) p
                WHERE m.type='table' AND m.name NOT LIKE 'sqlite_%'
                GROUP BY m.name
                HAVING COUNT(*) > 10000
            """)).fetchall()
            
            for (table,) in large_tables:
                suggestions.append(f"Table '{table}' is large - consider partitioning or archiving")
        
        except:
            suggestions.append("Unable to analyze query performance")
        
        return suggestions


class CachedQueries:
    """Cached query results for frequently accessed data."""
    
    _cache = {}
    _cache_timeout = 300  # 5 minutes
    
    @classmethod
    def get_dashboard_data(cls, session: Session, force_refresh: bool = False) -> Dict[str, Any]:
        """Get cached dashboard data."""
        cache_key = 'dashboard_data'
        now = datetime.now()
        
        # Check cache
        if not force_refresh and cache_key in cls._cache:
            cached_data, cache_time = cls._cache[cache_key]
            if (now - cache_time).seconds < cls._cache_timeout:
                return cached_data
        
        # Generate fresh data
        dashboard_data = {
            'plant_stats': PlantQueries.get_plants_with_planting_counts(session, limit=10),
            'recent_plantings': PlantingQueries.get_active_plantings_with_plants(session),
            'conversation_stats': ConversationQueries.get_conversations_with_message_counts(session, limit=5),
            'knowledge_stats': KnowledgeQueries.get_category_statistics(session),
            'generated_at': now.isoformat()
        }
        
        # Cache the data
        cls._cache[cache_key] = (dashboard_data, now)
        return dashboard_data
    
    @classmethod
    def clear_cache(cls):
        """Clear all cached data."""
        cls._cache.clear()


# Factory function for getting optimized queries
def get_queries(session: Session) -> Dict[str, Any]:
    """Get query utilities."""
    return {
        'plants': PlantQueries(),
        'plantings': PlantingQueries(),
        'conversations': ConversationQueries(),
        'knowledge': KnowledgeQueries(),
        'performance': PerformanceQueries(),
        'cached': CachedQueries()
    }