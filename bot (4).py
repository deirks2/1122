import discord
from discord.ext import commands
from google import genai
from google.genai import types
import os
import json
import asyncio
from collections import defaultdict, deque

# ── 환경변수 ──────────────────────────────────────────────
DISCORD_TOKEN   = os.environ["DISCORD_TOKEN"]
GEMINI_API_KEY  = os.environ["GEMINI_API_KEY"]
BOT_NAME        = os.environ.get("BOT_NAME", "봇")
ADMIN_ROLE_NAME = os.environ.get("ADMIN_ROLE", "관리자")
DATA_FILE       = "data.json"
MAX_HISTORY     = 30  # 기억할 최대 대화 횟수

# ── 새 Gemini SDK 초기화 ──────────────────────────────────
client_ai = genai.Client(api_key=GEMINI_API_KEY)

# ── 기본 시스템 프롬프트 ──────────────────────────────────
DEFAULT_SYSTEM_PROMPT = (
    f"너는 '{BOT_NAME}'라는 이름의 친절하고 유능한 디스코드 봇이야. "
    "한국어로 자연스럽게 대화해. 질문에 정확하고 도움이 되는 답변을 해줘."
)

# ── 데이터 로드/저장 ──────────────────────────────────────
def load_data() -> dict:
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"allowed_channels": [], "system_prompt": None}

def save_data(d: dict):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

data = load_data()
allowed_channels: set[int] = set(data.get("allowed_channels", []))

# data.json에 저장된 프롬프트가 있으면 사용, 없으면 기본값
if "system_prompt" not in data:
    data["system_prompt"] = None

def get_system_prompt() -> str:
    """현재 유효한 시스템 프롬프트 반환"""
    return data["system_prompt"] if data["system_prompt"] else DEFAULT_SYSTEM_PROMPT

# ── 대화 기록 (유저별) ────────────────────────────────────
history: dict[int, deque] = defaultdict(lambda: deque(maxlen=MAX_HISTORY * 2))

# ── Discord 봇 설정 ───────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)

# ── 권한 체크 ─────────────────────────────────────────────
def is_admin(member: discord.Member) -> bool:
    if member.guild_permissions.administrator:
        return True
    return any(r.name == ADMIN_ROLE_NAME for r in member.roles)

# ── Gemini 비동기 호출 헬퍼 ───────────────────────────────
async def call_gemini(user_id: int, user_message: str) -> str:
    hist = history[user_id]

    contents = []
    for turn in hist:
        contents.append(
            types.Content(
                role=turn["role"],
                parts=[types.Part(text=turn["parts"][0]["text"])]
            )
        )
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part(text=user_message)]
        )
    )

    config = types.GenerateContentConfig(
        system_instruction=get_system_prompt(),  # 항상 최신 프롬프트 사용
        max_output_tokens=2048,
    )

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client_ai.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=config,
        )
    )

    reply = response.text

    hist.append({"role": "user",  "parts": [{"text": user_message}]})
    hist.append({"role": "model", "parts": [{"text": reply}]})

    return reply

# ════════════════════════════════════════════════════════
#  이벤트
# ════════════════════════════════════════════════════════
@bot.event
async def on_ready():
    print(f"✅ {bot.user} 로그인 완료 | 모델: gemini-2.5-flash")

@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    await bot.process_commands(message)

    if message.content.startswith("!"):
        return

    if allowed_channels and message.channel.id not in allowed_channels:
        return

    mentioned = (
        bot.user.mentioned_in(message)
        or BOT_NAME.lower() in message.content.lower()
    )
    if not mentioned:
        return

    content = message.content
    content = content.replace(f"<@{bot.user.id}>", "").replace(f"<@!{bot.user.id}>", "")
    content = content.replace(BOT_NAME, "").strip()

    if not content:
        await message.channel.send("네, 무엇을 도와드릴까요? 😊")
        return

    try:
        async with message.channel.typing():
            reply = await call_gemini(message.author.id, content)

        for i in range(0, len(reply), 2000):
            await message.channel.send(reply[i:i+2000])

    except Exception as e:
        await message.channel.send(f"⚠️ Gemini 오류가 발생했어요: `{e}`")

# ════════════════════════════════════════════════════════
#  유저 명령어
# ════════════════════════════════════════════════════════
@bot.command(name="도움말")
async def help_cmd(ctx: commands.Context):
    embed = discord.Embed(title=f"📖 {BOT_NAME} 도움말", color=discord.Color.blurple())
    embed.add_field(
        name="💬 봇 대화 방법",
        value=(
            f"메시지에 **{BOT_NAME}** 을 포함하거나 @멘션하면 AI가 답변해요!\n"
            f"최근 **{MAX_HISTORY}번**의 대화를 기억합니다.\n"
            f"🤖 모델: `gemini-2.5-flash`"
        ),
        inline=False,
    )
    embed.add_field(name="👤 유저 명령어", value="`!도움말` — 이 메시지 표시", inline=False)
    embed.add_field(
        name="🔒 관리자 전용 (채널)",
        value=(
            "`!채널등록 [#채널]` — 봇 사용 채널 추가\n"
            "`!채널해제 [#채널]` — 봇 사용 채널 제거\n"
            "`!채널목록` — 허용된 채널 확인\n"
            "`!기록확인` — 대화 기록 개수 확인\n"
            "`!초기화` — 대화 기록 초기화"
        ),
        inline=False,
    )
    embed.add_field(
        name="🎭 관리자 전용 (봇 성격 설정)",
        value=(
            "`!성격보기` — 현재 봇 성격(시스템 프롬프트) 확인\n"
            "`!성격설정 [내용]` — 봇 성격 변경 (즉시 적용)\n"
            "`!성격추가 [내용]` — 현재 성격에 내용 추가\n"
            "`!성격초기화` — 봇 성격을 기본값으로 복원"
        ),
        inline=False,
    )
    embed.set_footer(text="관리자 명령어는 서버 관리자 권한 또는 '관리자' 역할 필요")
    await ctx.send(embed=embed)

# ════════════════════════════════════════════════════════
#  관리자 명령어 — 채널 관리
# ════════════════════════════════════════════════════════
@bot.command(name="채널등록")
async def add_channel(ctx: commands.Context, channel: discord.TextChannel = None):
    if not is_admin(ctx.author):
        return await ctx.send("❌ 관리자 권한이 필요합니다.")
    target = channel or ctx.channel
    if target.id in allowed_channels:
        return await ctx.send(f"ℹ️ {target.mention} 은(는) 이미 등록된 채널이에요.")
    allowed_channels.add(target.id)
    data["allowed_channels"] = list(allowed_channels)
    save_data(data)
    await ctx.send(f"✅ {target.mention} 채널을 봇 사용 채널로 등록했어요.")

@bot.command(name="채널해제")
async def remove_channel(ctx: commands.Context, channel: discord.TextChannel = None):
    if not is_admin(ctx.author):
        return await ctx.send("❌ 관리자 권한이 필요합니다.")
    target = channel or ctx.channel
    if target.id not in allowed_channels:
        return await ctx.send(f"ℹ️ {target.mention} 은(는) 등록되지 않은 채널이에요.")
    allowed_channels.discard(target.id)
    data["allowed_channels"] = list(allowed_channels)
    save_data(data)
    await ctx.send(f"✅ {target.mention} 채널을 봇 사용 채널에서 해제했어요.")

@bot.command(name="채널목록")
async def list_channels(ctx: commands.Context):
    if not is_admin(ctx.author):
        return await ctx.send("❌ 관리자 권한이 필요합니다.")
    if not allowed_channels:
        return await ctx.send("ℹ️ 등록된 채널 없음 — 현재 **모든 채널**에서 동작 중이에요.")
    mentions = []
    for cid in allowed_channels:
        ch = ctx.guild.get_channel(cid)
        mentions.append(ch.mention if ch else f"(삭제된 채널 ID:{cid})")
    await ctx.send("📋 **허용 채널 목록:**\n" + "\n".join(f"• {m}" for m in mentions))

@bot.command(name="기록확인")
async def check_history(ctx: commands.Context):
    if not is_admin(ctx.author):
        return await ctx.send("❌ 관리자 권한이 필요합니다.")
    count = len(history[ctx.author.id]) // 2
    await ctx.send(f"🗂️ {ctx.author.mention} 님의 대화 기록: **{count}번** / 최대 {MAX_HISTORY}번")

@bot.command(name="초기화")
async def reset_history(ctx: commands.Context):
    if not is_admin(ctx.author):
        return await ctx.send("❌ 관리자 권한이 필요합니다.")
    history[ctx.author.id].clear()
    await ctx.send(f"🗑️ {ctx.author.mention} 님의 대화 기록을 초기화했어요.")

# ════════════════════════════════════════════════════════
#  관리자 명령어 — 봇 성격(시스템 프롬프트) 관리
# ════════════════════════════════════════════════════════
@bot.command(name="성격보기")
async def show_prompt(ctx: commands.Context):
    """현재 설정된 시스템 프롬프트를 확인합니다."""
    if not is_admin(ctx.author):
        return await ctx.send("❌ 관리자 권한이 필요합니다.")

    current = get_system_prompt()
    is_custom = data["system_prompt"] is not None
    status = "✏️ **커스텀 설정**" if is_custom else "📌 **기본값**"

    embed = discord.Embed(
        title="🎭 현재 봇 성격 (시스템 프롬프트)",
        color=discord.Color.green() if is_custom else discord.Color.greyple()
    )
    embed.add_field(name="상태", value=status, inline=False)

    # 1024자 embed 제한 대응
    if len(current) <= 1000:
        embed.add_field(name="내용", value=f"```{current}```", inline=False)
        await ctx.send(embed=embed)
    else:
        await ctx.send(embed=embed)
        # 2000자씩 분할 전송
        for i in range(0, len(current), 1900):
            await ctx.send(f"```{current[i:i+1900]}```")


@bot.command(name="성격설정")
async def set_prompt(ctx: commands.Context, *, new_prompt: str = None):
    """봇의 시스템 프롬프트를 새로 설정합니다. Railway 재시작 후에도 유지됩니다."""
    if not is_admin(ctx.author):
        return await ctx.send("❌ 관리자 권한이 필요합니다.")

    if not new_prompt:
        return await ctx.send(
            "⚠️ 설정할 성격 내용을 입력해주세요.\n"
            "예시: `!성격설정 너는 냉소적이고 직설적인 봇이야. 짧고 간결하게 답해.`"
        )

    data["system_prompt"] = new_prompt
    save_data(data)

    # 성격이 바뀌었으므로 모든 대화 기록 초기화 (선택사항 — 일관성 유지)
    history.clear()

    embed = discord.Embed(
        title="✅ 봇 성격이 변경되었습니다",
        color=discord.Color.green()
    )
    preview = new_prompt if len(new_prompt) <= 500 else new_prompt[:500] + "..."
    embed.add_field(name="새 성격 (미리보기)", value=f"```{preview}```", inline=False)
    embed.set_footer(text="즉시 적용됨 | 대화 기록이 초기화되었습니다 | data.json에 저장됨")
    await ctx.send(embed=embed)


@bot.command(name="성격추가")
async def append_prompt(ctx: commands.Context, *, extra: str = None):
    """현재 시스템 프롬프트 뒤에 내용을 추가합니다."""
    if not is_admin(ctx.author):
        return await ctx.send("❌ 관리자 권한이 필요합니다.")

    if not extra:
        return await ctx.send(
            "⚠️ 추가할 내용을 입력해주세요.\n"
            "예시: `!성격추가 절대로 이모티콘을 사용하지 마.`"
        )

    current = get_system_prompt()
    updated = current + "\n" + extra
    data["system_prompt"] = updated
    save_data(data)

    history.clear()

    embed = discord.Embed(
        title="✅ 봇 성격에 내용이 추가되었습니다",
        color=discord.Color.blue()
    )
    embed.add_field(name="추가된 내용", value=f"```{extra}```", inline=False)
    total_len = len(updated)
    embed.set_footer(text=f"전체 프롬프트 길이: {total_len}자 | 즉시 적용됨 | 대화 기록 초기화됨")
    await ctx.send(embed=embed)


@bot.command(name="성격초기화")
async def reset_prompt(ctx: commands.Context):
    """시스템 프롬프트를 기본값으로 복원합니다."""
    if not is_admin(ctx.author):
        return await ctx.send("❌ 관리자 권한이 필요합니다.")

    data["system_prompt"] = None
    save_data(data)
    history.clear()

    embed = discord.Embed(
        title="🔄 봇 성격이 기본값으로 초기화되었습니다",
        color=discord.Color.orange()
    )
    embed.add_field(name="기본 성격", value=f"```{DEFAULT_SYSTEM_PROMPT}```", inline=False)
    embed.set_footer(text="즉시 적용됨 | 대화 기록이 초기화되었습니다")
    await ctx.send(embed=embed)

# ── 실행 ──────────────────────────────────────────────────
bot.run(DISCORD_TOKEN)
