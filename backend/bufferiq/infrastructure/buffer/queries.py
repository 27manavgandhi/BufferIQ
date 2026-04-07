"""
GraphQL query and mutation definitions for Buffer API.
"""

# Organizations
GET_ORGANIZATIONS = """
query GetOrganizations {
    organizations {
        id
        name
        createdAt
    }
}
"""

GET_ORGANIZATION = """
query GetOrganization($id: ID!) {
    organization(id: $id) {
        id
        name
        createdAt
    }
}
"""

# Channels
GET_CHANNELS = """
query GetChannels($organizationId: ID!) {
    channels(organizationId: $organizationId) {
        id
        organizationId
        platform
        handle
        isActive
        createdAt
    }
}
"""

GET_CHANNEL = """
query GetChannel($id: ID!) {
    channel(id: $id) {
        id
        organizationId
        platform
        handle
        isActive
        createdAt
    }
}
"""

# Posts
GET_POSTS = """
query GetPosts($channelId: ID!, $limit: Int, $offset: Int) {
    posts(channelId: $channelId, limit: $limit, offset: $offset) {
        id
        channelId
        content
        status
        scheduledAt
        sentAt
        engagement {
            likes
            comments
            shares
            impressions
            clicks
        }
        createdAt
        updatedAt
    }
}
"""

GET_POST = """
query GetPost($id: ID!) {
    post(id: $id) {
        id
        channelId
        content
        status
        scheduledAt
        sentAt
        engagement {
            likes
            comments
            shares
            impressions
            clicks
        }
        createdAt
        updatedAt
    }
}
"""

# Mutations
CREATE_POST = """
mutation CreatePost($channelId: ID!, $content: String!, $scheduledAt: String) {
    createPost(channelId: $channelId, content: $content, scheduledAt: $scheduledAt) {
        id
        channelId
        content
        status
        scheduledAt
        createdAt
    }
}
"""

UPDATE_POST = """
mutation UpdatePost($id: ID!, $content: String, $scheduledAt: String) {
    updatePost(id: $id, content: $content, scheduledAt: $scheduledAt) {
        id
        content
        scheduledAt
        updatedAt
    }
}
"""

DELETE_POST = """
mutation DeletePost($id: ID!) {
    deletePost(id: $id) {
        success
    }
}
"""
