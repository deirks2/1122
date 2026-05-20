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
# ✅ google-generativeai (구버전) → google-genai (신버전)
client_ai = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = (
    f"너는 '{BOT_NAME}'라는 이름의 친절하고 유능한 디스코드 봇이야. "
    "한국어로 자연스럽게 대화해. 질문에 정확하고 도움이 되는 답변을 해줘."
)

# ── 데이터 로드/저장 ──────────────────────────────────────
def load_data() -> dict:
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"allowed_channels": []}

def save_data(d: dict):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

data = load_data()
allowed_channels: set[int] = set(data.get("allowed_channels", []))

# ── 대화 기록 (유저별) ────────────────────────────────────
# 새 SDK는 Contents 리스트로 히스토리 관리
# [{"role": "user", "parts": [{"text": "..."}]}, {"role": "model", "parts": [{"text": "..."}]}]
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

    # 히스토리 → types.Content 리스트로 변환
    contents = []
    for turn in hist:
        contents.append(
            types.Content(
                role=turn["role"],
                parts=[types.Part(text=turn["parts"][0]["text"])]
            )
        )
    # 현재 메시지 추가
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part(text=user_message)]
        )
    )

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=2048,
    )

    # 동기 API를 스레드풀에서 실행 (Discord 비동기 루프 블로킹 방지)
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

    # 기록 저장
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

    await bot.process_commands(message)  # 명령어 우선

    if message.content.startswith("!"):
        return

    # 허용 채널 체크 (없으면 전체 허용)
    if allowed_channels and message.channel.id not in allowed_channels:
        return

    # 봇 이름 or @멘션 감지
    mentioned = (
        bot.user.mentioned_in(message)
        or BOT_NAME.lower() in message.content.lower()
    )
    if not mentioned:
        return

    # 봇 이름·멘션 제거 후 순수 질문 추출
    content = message.content
    content = content.replace(f"<@{bot.user.id}>", "").replace(f"<@!{bot.user.id}>", "")
    content = content.replace(BOT_NAME, "").strip()

    if not content:
        await message.channel.send("네, 무엇을 도와드릴까요? 😊")
        return

    try:
        async with message.channel.typing():
            reply = await call_gemini(message.author.id, content)

        # 2000자 제한 분할 전송
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
        name="🔒 관리자 전용",
        value=(
            "`!채널등록 [#채널]` — 봇 사용 채널 추가\n"
            "`!채널해제 [#채널]` — 봇 사용 채널 제거\n"
            "`!채널목록` — 허용된 채널 확인\n"
            "`!기록확인` — 대화 기록 개수 확인\n"
            "`!초기화` — 대화 기록 초기화"
        ),
        inline=False,
    )
    embed.set_footer(text="관리자 명령어는 서버 관리자 권한 또는 '관리자' 역할 필요")
    await ctx.send(embed=embed)

# ════════════════════════════════════════════════════════
#  관리자 명령어
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

# ── 실행 ──────────────────────────────────────────────────
bot.run(DISCORD_TOKEN)
