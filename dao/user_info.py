from tortoise import fields
from tortoise.models import Model
from datetime import datetime
from typing import Optional


class UserInfo(Model):
    """用户信息表"""

    # ID
    id = fields.BigIntField(pk=True, description='ID')

    session_id = fields.CharField(
        null=False,
        max_length=36,
        default='',
        description='session_id'
    )

    # 业务字段
    name = fields.CharField(
        max_length=32,
        default='sys',
        description='姓名'
    )
    birthday = fields.DatetimeField(
        null=True,
        description='出生日期'
    )
    gender = fields.IntField(
        default=1,
        description='性别: 0女 1男'
    )
    hobby = fields.CharField(
        max_length=255,
        default='',
        description='兴趣爱好(多个用英文逗号分隔)'
    )
    skill = fields.CharField(
        max_length=255,
        default='',
        description='技能(多个用英文逗号分隔)'
    )

    # 常规字段
    created_time = fields.DatetimeField(
        auto_now_add=True,
        description='创建时间'
    )
    updated_time = fields.DatetimeField(
        auto_now=True,
        description='更新时间'
    )

    class Meta:
        table = "user_info"
        table_description = "用户信息表"

    def __str__(self):
        return f"UserInfo(id={self.id}, name={self.name})"

    # 辅助方法：将爱好列表转换为字符串
    def set_hobbies(self, hobbies: list[str]):
        """设置兴趣爱好（从列表转为逗号分隔字符串）"""
        self.hobby = ','.join(hobbies)

    # 辅助方法：将字符串转换为爱好列表
    def get_hobbies(self) -> list[str]:
        """获取兴趣爱好列表"""
        return [h.strip() for h in self.hobby.split(',') if h.strip()]

    # 辅助方法：将技能列表转换为字符串
    def set_skills(self, skills: list[str]):
        """设置技能（从列表转为逗号分隔字符串）"""
        self.skill = ','.join(skills)

    # 辅助方法：将字符串转换为技能列表
    def get_skills(self) -> list[str]:
        """获取技能列表"""
        return [s.strip() for s in self.skill.split(',') if s.strip()]