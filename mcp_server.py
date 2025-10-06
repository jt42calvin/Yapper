"""
MCP Server for Laugh Line - Comedy Clip Finder
Provides programmatic access to cached videos, transcripts, and feedback data

Install: pip install mcp

Run standalone: python mcp_server.py
Or it will auto-start when running laugh_line.py
"""

import json
import os
from pathlib import Path
from typing import Any
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp import types

# Directories
DOWNLOADS_DIR = "downloads"
TRANSCRIPTS_DIR = "transcripts"
HIGHLIGHTS_DIR = "highlights"
FEEDBACK_DIR = "feedback"

# Initialize MCP server
app = Server("laugh-line-mcp")

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    """List all available MCP tools"""
    return [
        types.Tool(
            name="list_cached_videos",
            description="List all cached videos with their IDs and metadata",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="get_video_transcript",
            description="Get the transcript for a specific video by video ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_id": {
                        "type": "string",
                        "description": "YouTube video ID"
                    }
                },
                "required": ["video_id"]
            }
        ),
        types.Tool(
            name="get_feedback_history",
            description="Get feedback history for a specific video",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_id": {
                        "type": "string",
                        "description": "YouTube video ID"
                    }
                },
                "required": ["video_id"]
            }
        ),
        types.Tool(
            name="get_video_info",
            description="Get metadata about cached videos and their highlights",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_id": {
                        "type": "string",
                        "description": "YouTube video ID (optional - if not provided, lists all)"
                    }
                },
                "required": []
            }
        ),
        types.Tool(
            name="search_transcript",
            description="Search for specific text within a video's transcript",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_id": {
                        "type": "string",
                        "description": "YouTube video ID"
                    },
                    "query": {
                        "type": "string",
                        "description": "Text to search for in the transcript"
                    }
                },
                "required": ["video_id", "query"]
            }
        ),
        types.Tool(
            name="get_flagged_sections",
            description="Get all flagged (inappropriate or not funny) sections across all videos",
            inputSchema={
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Filter by reason: 'not_appropriate' or 'not_funny' (optional)"
                    }
                },
                "required": []
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[types.TextContent]:
    """Handle tool calls"""
    
    if name == "list_cached_videos":
        return await list_cached_videos()
    
    elif name == "get_video_transcript":
        video_id = arguments.get("video_id")
        return await get_video_transcript(video_id)
    
    elif name == "get_feedback_history":
        video_id = arguments.get("video_id")
        return await get_feedback_history(video_id)
    
    elif name == "get_video_info":
        video_id = arguments.get("video_id")
        return await get_video_info(video_id)
    
    elif name == "search_transcript":
        video_id = arguments.get("video_id")
        query = arguments.get("query")
        return await search_transcript(video_id, query)
    
    elif name == "get_flagged_sections":
        reason = arguments.get("reason")
        return await get_flagged_sections(reason)
    
    else:
        raise ValueError(f"Unknown tool: {name}")

# Tool implementations

async def list_cached_videos() -> list[types.TextContent]:
    """List all cached videos"""
    videos = []
    
    if os.path.exists(DOWNLOADS_DIR):
        for filename in os.listdir(DOWNLOADS_DIR):
            if filename.endswith("_video.mp4"):
                video_id = filename.replace("_video.mp4", "")
                
                # Check if transcript exists
                transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{video_id}_transcript.txt")
                has_transcript = os.path.exists(transcript_path)
                
                # Check if highlight exists
                highlight_path = os.path.join(HIGHLIGHTS_DIR, f"{video_id}_highlight.mp4")
                has_highlight = os.path.exists(highlight_path)
                
                # Check if feedback exists
                feedback_path = os.path.join(FEEDBACK_DIR, f"{video_id}_feedback.json")
                has_feedback = os.path.exists(feedback_path)
                
                videos.append({
                    "video_id": video_id,
                    "has_transcript": has_transcript,
                    "has_highlight": has_highlight,
                    "has_feedback": has_feedback
                })
    
    result = {
        "total_videos": len(videos),
        "videos": videos
    }
    
    return [types.TextContent(
        type="text",
        text=json.dumps(result, indent=2)
    )]

async def get_video_transcript(video_id: str) -> list[types.TextContent]:
    """Get transcript for a specific video"""
    transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{video_id}_transcript.txt")
    
    if not os.path.exists(transcript_path):
        return [types.TextContent(
            type="text",
            text=f"Error: Transcript not found for video ID: {video_id}"
        )]
    
    with open(transcript_path, 'r') as f:
        transcript_text = f.read()
    
    result = {
        "video_id": video_id,
        "transcript": transcript_text
    }
    
    return [types.TextContent(
        type="text",
        text=json.dumps(result, indent=2)
    )]

async def get_feedback_history(video_id: str) -> list[types.TextContent]:
    """Get feedback history for a video"""
    feedback_path = os.path.join(FEEDBACK_DIR, f"{video_id}_feedback.json")
    
    if not os.path.exists(feedback_path):
        return [types.TextContent(
            type="text",
            text=f"No feedback found for video ID: {video_id}"
        )]
    
    with open(feedback_path, 'r') as f:
        feedback_data = json.load(f)
    
    return [types.TextContent(
        type="text",
        text=json.dumps(feedback_data, indent=2)
    )]

async def get_video_info(video_id: str = None) -> list[types.TextContent]:
    """Get metadata about videos"""
    if video_id:
        # Get info for specific video
        video_path = os.path.join(DOWNLOADS_DIR, f"{video_id}_video.mp4")
        if not os.path.exists(video_path):
            return [types.TextContent(
                type="text",
                text=f"Error: Video not found for ID: {video_id}"
            )]
        
        info = {
            "video_id": video_id,
            "video_path": video_path,
            "video_size_mb": round(os.path.getsize(video_path) / (1024 * 1024), 2)
        }
        
        # Add transcript info
        transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{video_id}_transcript.txt")
        if os.path.exists(transcript_path):
            with open(transcript_path, 'r') as f:
                lines = f.readlines()
            info["transcript_lines"] = len(lines)
        
        # Add highlight info
        highlight_path = os.path.join(HIGHLIGHTS_DIR, f"{video_id}_highlight.mp4")
        if os.path.exists(highlight_path):
            info["highlight_path"] = highlight_path
            info["highlight_size_mb"] = round(os.path.getsize(highlight_path) / (1024 * 1024), 2)
        
        return [types.TextContent(
            type="text",
            text=json.dumps(info, indent=2)
        )]
    else:
        # List all videos with basic info
        return await list_cached_videos()

async def search_transcript(video_id: str, query: str) -> list[types.TextContent]:
    """Search for text in a transcript"""
    transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{video_id}_transcript.txt")
    
    if not os.path.exists(transcript_path):
        return [types.TextContent(
            type="text",
            text=f"Error: Transcript not found for video ID: {video_id}"
        )]
    
    matches = []
    with open(transcript_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if query.lower() in line.lower():
                matches.append({
                    "line_number": line_num,
                    "content": line.strip()
                })
    
    result = {
        "video_id": video_id,
        "query": query,
        "matches_found": len(matches),
        "matches": matches
    }
    
    return [types.TextContent(
        type="text",
        text=json.dumps(result, indent=2)
    )]

async def get_flagged_sections(reason: str = None) -> list[types.TextContent]:
    """Get all flagged sections across all videos"""
    flagged_sections = []
    
    if os.path.exists(FEEDBACK_DIR):
        for filename in os.listdir(FEEDBACK_DIR):
            if filename.endswith("_feedback.json"):
                with open(os.path.join(FEEDBACK_DIR, filename), 'r') as f:
                    feedback_data = json.load(f)
                
                for section in feedback_data.get("flagged_sections", []):
                    if reason is None or section.get("reason") == reason:
                        flagged_sections.append({
                            "video_id": feedback_data.get("video_id"),
                            "start": section.get("start"),
                            "end": section.get("end"),
                            "reason": section.get("reason")
                        })
    
    result = {
        "total_flagged": len(flagged_sections),
        "filter_reason": reason if reason else "all",
        "flagged_sections": flagged_sections
    }
    
    return [types.TextContent(
        type="text",
        text=json.dumps(result, indent=2)
    )]

async def main():
    """Run the MCP server"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="laugh-line-mcp",
                server_version="1.0.0",
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
